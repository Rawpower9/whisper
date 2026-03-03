#!/usr/bin/env python3
"""
whisper_type.py — Hold Right Option to record, releases transcribes & types result.
"""

import logging
import os
import re
import subprocess
import threading

import ollama
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
STREAM_STEP_MS = 1000  # interval between partial transcriptions when REALTIME=True

MODELS = ["tiny", "base", "small", "medium", "large-v3"]
DEFAULT_MODEL = "large-v3"

MIN_SPEECH_RMS = 0.005   # below this → likely silence; skip whisper to avoid hallucinations
TARGET_RMS     = 0.1     # normalise speech to this level before passing to whisper

LLM_ENABLED = True
LLM_MODEL   = "qwen3:4b"
LLM_SYSTEM_PROMPT = (
    "You are a transcription editor. Clean up the raw speech-to-text output below:\n"
    "- Fix grammar, punctuation, and capitalization\n"
    "- Remove filler words (um, uh, like, you know, so, right)\n"
    "- Fix obvious transcription errors using context\n"
    "- Keep the original meaning and intent intact\n"
    "Output ONLY the cleaned text. No explanation, no quotes, nothing else."
)


def model_path(name):
    return os.path.join(MODELS_DIR, f"ggml-{name}.bin")


def _prepare_audio(chunks: list) -> "np.ndarray | None":
    """Concatenate raw int16 chunks, check energy, and normalise for Whisper.

    Returns None (skip Whisper) when the recording is too quiet to be speech,
    which prevents Whisper's common hallucinations on near-silence.
    Otherwise returns float32 audio normalised to TARGET_RMS.
    """
    audio = np.concatenate(chunks).astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(audio ** 2)))
    log.debug("Audio RMS: %.5f", rms)
    if rms < MIN_SPEECH_RMS:
        log.debug("Audio too quiet (RMS %.5f < %.5f); skipping whisper", rms, MIN_SPEECH_RMS)
        return None
    audio = audio * (TARGET_RMS / rms)
    return np.clip(audio, -1.0, 1.0)


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            n += 1
        else:
            break
    return n


def _llm_refine(text: str) -> str:
    """Pass raw whisper text through a local LLM for cleanup."""
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            think=False,
        )
        refined = response["message"]["content"].strip()
        refined = re.sub(r"<think>.*?</think>", "", refined, flags=re.DOTALL).strip()
        log.debug("LLM: %r -> %r", text, refined)
        return refined if refined else text
    except Exception:
        log.exception("LLM refinement failed; using raw whisper text")
        return text


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
        self._typed_text = ""        # chars typed to screen this session
        self._user_interacted = False  # user pressed a key mid-session

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

    def _on_press(self, key, injected=False):
        if injected:
            return  # ignore our own simulated keystrokes

        if key != keyboard.Key.alt_r:
            with self._lock:
                if self._state == "recording":
                    self._user_interacted = True  # cursor may have moved
            return

        with self._lock:
            if self._state != "idle":
                return
            self._state = "recording"
            self._audio_chunks = []
            self._audio_done.clear()
            self._typed_text = ""           # reset for new session
            self._user_interacted = False   # reset for new session

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
    # In-place text application (shared by partial and final)
    # ------------------------------------------------------------------

    def _apply_text(self, new_text):
        with self._lock:
            old_text = self._typed_text
            interacted = self._user_interacted
            if interacted:
                self._user_interacted = False

        if interacted:
            log.debug("User interacted; skipping update and resetting typed tracking")
            with self._lock:
                self._typed_text = ""
            return

        common_len = _common_prefix_len(old_text, new_text)
        n_back = len(old_text) - common_len
        new_suffix = new_text[common_len:]

        log.debug("apply_text: back=%d suffix=%r", n_back, new_suffix)

        for _ in range(n_back):
            self._typer.tap(keyboard.Key.backspace)
        if new_suffix:
            self._typer.type(new_suffix)

        with self._lock:
            self._typed_text = new_text

    # ------------------------------------------------------------------
    # Partial transcription (REALTIME=True only)
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
                audio = _prepare_audio(chunks)
                if audio is None:
                    return
                segments = self._whisper.transcribe(audio)
                text = "".join(seg.text for seg in segments).strip()
                text = re.sub(r"\[.*?\]|\(.*?\)", "", text).strip()
                log.debug("Partial: %r", text)
                if text:
                    self._apply_text(text)
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

            audio = _prepare_audio(chunks)
            if audio is None:
                log.debug("Audio too quiet; nothing to transcribe")
                return
            log.debug("Transcribing %.1f s of audio", len(audio) / SAMPLERATE)

            with self._whisper_lock:
                segments = self._whisper.transcribe(audio)

            text = "".join(seg.text for seg in segments).strip()
            text = re.sub(r"\[.*?\]|\(.*?\)", "", text).strip()
            log.debug("Whisper raw: %r", text)

            if LLM_ENABLED and text:
                text = _llm_refine(text)
            log.debug("Final text: %r", text)

            if text:
                self._apply_text(text)
            else:
                log.debug("No text to type")
        except Exception:
            log.exception("Error during transcription")
        finally:
            with self._lock:
                self._typed_text = ""
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
