#!/usr/bin/env python3
"""
whisper_type.py — Hold Right Option to record, releases transcribes & types result.
"""

import logging
import os
import re
import subprocess
import threading

import numpy as np
import rumps
from pywhispercpp.model import Model
from pynput import keyboard

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SAMPLERATE = 16000
CHANNELS = 1
MODELS_DIR = os.path.expanduser("~/.whisper/models")

# Set to True to also transcribe periodically while recording (partial results
# are logged). Both modes type the final result on release.
REALTIME = True
STREAM_STEP_MS = 3000  # interval between partial transcriptions when REALTIME=True

MODELS = ["tiny", "base", "small", "medium", "large-v3"]
DEFAULT_MODEL = "large-v3"


def model_path(name):
    return os.path.join(MODELS_DIR, f"ggml-{name}.bin")


class WhisperTypeApp(rumps.App):
    def __init__(self, initial_model):
        super().__init__("◉")
        self.menu = ["Quit"]

        self._lock = threading.Lock()
        self._whisper_lock = threading.Lock()  # serialise transcribe() calls
        self._state = "idle"
        self._flash = False
        self._rec_proc = None
        self._audio_chunks = []
        self._audio_done = threading.Event()
        self._stream_timer = None

        self._typer = keyboard.Controller()

        # Load model once at startup.
        # Metal (Apple GPU) is used automatically when pywhispercpp is built
        # with Metal support, which is the default via brew whisper-cpp.
        self._whisper = self._load_model(initial_model)

        self._timer = rumps.Timer(self._update_ui, 0.4)
        self._timer.start()

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def _load_model(self, name):
        mpath = model_path(name)
        log.info("Loading model %s from %s ...", name, mpath)
        model = Model(mpath, n_threads=os.cpu_count() or 4)
        log.info("Model %s ready", name)
        return model

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _update_ui(self, _):
        with self._lock:
            state = self._state
            self._flash = not self._flash
            flash = self._flash

        if state == "idle":
            self.title = "◉"
        elif state == "recording":
            self.title = "⏺" if flash else "◌"
        else:  # transcribing
            self.title = "◌"

    # ------------------------------------------------------------------
    # Key events (pynput listener thread)
    # ------------------------------------------------------------------

    def _on_press(self, key):
        if key != keyboard.Key.alt_r:
            return

        with self._lock:
            if self._state != "idle":
                return
            self._state = "recording"
            self._audio_chunks = []
            self._audio_done.clear()

        log.debug("Recording started")

        proc = subprocess.Popen(
            [
                "rec",
                "-r", str(SAMPLERATE),
                "-c", str(CHANNELS),
                "-e", "signed-integer",
                "-b", "16",
                "-t", "raw",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        threading.Thread(target=self._read_audio, args=(proc,), daemon=True).start()

        if REALTIME:
            self._schedule_partial()

        with self._lock:
            self._rec_proc = proc

    def _on_release(self, key):
        if key != keyboard.Key.alt_r:
            return

        with self._lock:
            if self._state != "recording":
                return
            self._state = "transcribing"
            proc = self._rec_proc
            self._rec_proc = None

        if self._stream_timer is not None:
            self._stream_timer.cancel()
            self._stream_timer = None

        if proc is not None:
            proc.terminate()
            proc.wait()

        threading.Thread(target=self._transcribe, daemon=True).start()

    # ------------------------------------------------------------------
    # Audio capture
    # ------------------------------------------------------------------

    def _read_audio(self, proc):
        try:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                samples = np.frombuffer(chunk, dtype=np.int16).copy()
                with self._lock:
                    self._audio_chunks.append(samples)
        finally:
            self._audio_done.set()
            log.debug("Audio reader done")

    # ------------------------------------------------------------------
    # Partial transcription (REALTIME=True only — logged, not typed)
    # ------------------------------------------------------------------

    def _schedule_partial(self):
        self._stream_timer = threading.Timer(STREAM_STEP_MS / 1000.0, self._do_partial)
        self._stream_timer.daemon = True
        self._stream_timer.start()

    def _do_partial(self):
        with self._lock:
            if self._state != "recording":
                return
            chunks = list(self._audio_chunks)

        if chunks and self._whisper_lock.acquire(blocking=False):
            try:
                audio = np.concatenate(chunks).astype(np.float32) / 32768.0
                segments = self._whisper.transcribe(audio)
                text = "".join(seg.text for seg in segments).strip()
                text = re.sub(r"\[.*?\]|\(.*?\)", "", text).strip()
                log.debug("Partial: %r", text)
            except Exception:
                log.exception("Error in partial transcription")
            finally:
                self._whisper_lock.release()

        with self._lock:
            if self._state == "recording":
                self._schedule_partial()

    # ------------------------------------------------------------------
    # Final transcription (daemon thread)
    # ------------------------------------------------------------------

    def _transcribe(self):
        try:
            # Wait for the audio reader to flush any remaining buffered bytes
            self._audio_done.wait(timeout=2.0)

            with self._lock:
                chunks = list(self._audio_chunks)

            if not chunks:
                log.debug("No audio captured")
                return

            audio = np.concatenate(chunks).astype(np.float32) / 32768.0
            log.debug("Transcribing %.1f s of audio", len(audio) / SAMPLERATE)

            with self._whisper_lock:
                segments = self._whisper.transcribe(audio)

            text = "".join(seg.text for seg in segments).strip()
            text = re.sub(r"\[.*?\]|\(.*?\)", "", text).strip()
            log.debug("Final text: %r", text)

            if text:
                self._typer.type(text)
            else:
                log.debug("No text to type")
        except Exception:
            log.exception("Error during transcription")
        finally:
            with self._lock:
                self._state = "idle"

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    @rumps.clicked("Quit")
    def quit_app(self, _):
        self._listener.stop()
        rumps.quit_application()


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)

    initial = None
    if os.path.isfile(model_path(DEFAULT_MODEL)):
        initial = DEFAULT_MODEL
    else:
        for name in MODELS:
            if os.path.isfile(model_path(name)):
                initial = name
                break

    if initial is None:
        print("No models found in ~/.whisper/models/")
        print("Run: bash setup.sh")
        raise SystemExit(1)

    log.info("Starting with model: %s", initial)
    WhisperTypeApp(initial).run()
