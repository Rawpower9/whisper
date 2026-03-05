"""Main WhisperType menu bar application."""

import logging
import os
import subprocess
import threading

import numpy as np
import rumps
from pynput import keyboard

from .config import (
    SAMPLERATE, CHANNELS, REALTIME, STREAM_STEP_MS, MODELS,
    CHUNK_SAMPLES, LLM_ENABLED, SPEAKER_SIM_THRESHOLD,
    VOICE_PROFILE_PATH, ENROLL_MIN_SECONDS,
)
from .audio import prepare_audio, speaker_similarity, enroll_voice
from .transcription import transcribe, llm_refine
from .overlay import TranscriptionOverlay

log = logging.getLogger(__name__)


class WhisperTypeApp(rumps.App):
    def __init__(self, initial_model):
        super().__init__("◉")
        self.menu = ["Enroll My Voice", "Clear Voice Profile", None, "Quit"]

        self._lock = threading.Lock()
        self._whisper_lock = threading.Lock()
        self._state = "idle"
        self._flash = False
        self._rec_proc = None
        self._audio_chunks = []
        self._audio_done = threading.Event()
        self._stream_timer = None
        self._enrolling = False
        self._session = 0
        self._finalized_samples = 0
        self._partial_texts = []

        self._typer = keyboard.Controller()
        self._model_repo = MODELS.get(initial_model, initial_model)
        log.info("Using mlx-whisper model: %s (Metal GPU)", self._model_repo)

        self._overlay = TranscriptionOverlay()

        self._timer = rumps.Timer(self._update_ui, 0.4)
        self._timer.start()

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    # ── UI ────────────────────────────────────────────────────────────

    def _update_ui(self, _):
        with self._lock:
            state = self._state
            self._flash = not self._flash
            flash = self._flash

        if state == "idle":
            self.title = "◉"
        elif state == "recording":
            self.title = "⏺" if flash else "◌"
        else:
            self.title = "◌"

    # ── Key events ────────────────────────────────────────────────────

    def _on_press(self, key, injected=False):
        if injected:
            return
        if key != keyboard.Key.alt_r:
            return

        with self._lock:
            if self._state not in ("idle", "transcribing"):
                return
            self._session += 1
            self._state = "recording"
            self._audio_chunks = []
            self._audio_done = threading.Event()
            self._finalized_samples = 0
            self._partial_texts = []

        self._overlay.show()
        log.debug("Recording started (session %d)", self._session)

        proc = subprocess.Popen(
            [
                "rec",
                "-r", str(SAMPLERATE),
                "-c", str(CHANNELS),
                "-e", "signed-integer",
                "-b", "16",
                "-t", "raw",
                "-",
                "highpass", "80",
                "lowpass", "8000",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        with self._lock:
            done_event = self._audio_done

        threading.Thread(target=self._read_audio, args=(proc, done_event), daemon=True).start()

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
            session = self._session
            done_event = self._audio_done

        if self._stream_timer is not None:
            self._stream_timer.cancel()
            self._stream_timer = None

        if proc is not None:
            proc.terminate()
            proc.wait()

        threading.Thread(target=self._transcribe, args=(session, done_event), daemon=True).start()

    # ── Audio capture ─────────────────────────────────────────────────

    def _read_audio(self, proc, done_event):
        try:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                samples = np.frombuffer(chunk, dtype=np.int16).copy()
                with self._lock:
                    self._audio_chunks.append(samples)
        finally:
            done_event.set()
            log.debug("Audio reader done")

    # ── Partial transcription (REALTIME only) ─────────────────────────

    def _schedule_partial(self):
        self._stream_timer = threading.Timer(STREAM_STEP_MS / 1000.0, self._do_partial)
        self._stream_timer.daemon = True
        self._stream_timer.start()

    def _do_partial(self):
        with self._lock:
            if self._state != "recording":
                return
            chunks = list(self._audio_chunks)
            finalized_samples = self._finalized_samples
            partial_texts = list(self._partial_texts)

        if not chunks:
            with self._lock:
                if self._state == "recording":
                    self._schedule_partial()
            return

        if not self._whisper_lock.acquire(blocking=False):
            with self._lock:
                if self._state == "recording":
                    self._schedule_partial()
            return

        try:
            all_audio = np.concatenate(chunks)
            total_samples = len(all_audio)

            while total_samples - finalized_samples >= CHUNK_SAMPLES:
                chunk_audio = all_audio[finalized_samples:finalized_samples + CHUNK_SAMPLES]
                audio = prepare_audio(chunk_audio)
                if audio is not None:
                    text = transcribe(audio, self._model_repo, partial=True)
                    if text:
                        partial_texts.append(text)
                finalized_samples += CHUNK_SAMPLES

            tail_text = ""
            tail = all_audio[finalized_samples:]
            if len(tail) > SAMPLERATE // 2:
                audio = prepare_audio(tail)
                if audio is not None:
                    tail_text = transcribe(audio, self._model_repo, partial=True)

            with self._lock:
                self._finalized_samples = finalized_samples
                self._partial_texts = partial_texts

            parts = partial_texts + ([tail_text] if tail_text else [])
            full_text = " ".join(parts).strip()
            log.debug("Partial: %r", full_text)
            if full_text:
                self._overlay.update_text(full_text)
        except Exception:
            log.exception("Error in partial transcription")
        finally:
            self._whisper_lock.release()

        with self._lock:
            if self._state == "recording":
                self._schedule_partial()

    # ── Final transcription ───────────────────────────────────────────

    def _transcribe(self, session, done_event):
        try:
            done_event.wait(timeout=2.0)

            with self._lock:
                if self._session != session:
                    log.debug("Session %d superseded; discarding", session)
                    return
                chunks = list(self._audio_chunks)
                enrolling = self._enrolling
                self._enrolling = False

            if not chunks:
                log.debug("No audio captured")
                return

            all_audio = np.concatenate(chunks)
            audio = prepare_audio(all_audio)
            if audio is None:
                log.debug("Audio too quiet; nothing to transcribe")
                return

            if enrolling:
                self._do_enrollment(audio)
                return

            sim = speaker_similarity(audio)
            if sim is not None and sim < SPEAKER_SIM_THRESHOLD:
                log.debug("Speaker rejected (%.3f < %.2f)", sim, SPEAKER_SIM_THRESHOLD)
                return

            log.debug("Transcribing %.1f s of audio", len(audio) / SAMPLERATE)

            with self._whisper_lock:
                text = transcribe(audio, self._model_repo)
            log.debug("Whisper raw: %r", text)

            if LLM_ENABLED and text:
                text = llm_refine(text)
            log.debug("Final text: %r", text)

            with self._lock:
                if self._session != session:
                    log.debug("Session %d superseded after transcription; discarding", session)
                    return

            self._overlay.hide()
            if text:
                self._typer.type(text)
            else:
                log.debug("No text to type")
        except Exception:
            log.exception("Error during transcription")
        finally:
            with self._lock:
                if self._session == session:
                    self._state = "idle"
            self._overlay.hide()

    def _do_enrollment(self, audio):
        duration = len(audio) / SAMPLERATE
        if duration < ENROLL_MIN_SECONDS:
            rumps.notification("Enrollment Failed", "",
                f"Recording was {duration:.1f}s — hold for at least {ENROLL_MIN_SECONDS:.0f}s.")
            return
        embedding = enroll_voice(audio)
        if embedding is None:
            rumps.notification("Enrollment Failed", "", "Could not load voice encoder.")
            return
        os.makedirs(os.path.dirname(VOICE_PROFILE_PATH), exist_ok=True)
        np.save(VOICE_PROFILE_PATH, embedding)
        log.info("Voice profile saved to %s", VOICE_PROFILE_PATH)
        rumps.notification("Enrollment Complete", "",
            "Your voice profile has been saved. Speaker verification is now active.")

    # ── Menu actions ──────────────────────────────────────────────────

    @rumps.clicked("Enroll My Voice")
    def enroll_voice_menu(self, _):
        with self._lock:
            if self._state != "idle":
                rumps.alert("Cannot Enroll", message="Wait until the app is idle first.")
                return
            self._enrolling = True
        rumps.alert("Voice Enrollment",
            message="Hold Right Option and speak naturally for at least 3 seconds, then release.")

    @rumps.clicked("Clear Voice Profile")
    def clear_voice_profile(self, _):
        if os.path.isfile(VOICE_PROFILE_PATH):
            os.remove(VOICE_PROFILE_PATH)
            log.info("Voice profile removed")
            rumps.alert("Voice Profile Cleared",
                message="Speaker verification is now disabled. Re-enroll to re-activate.")
        else:
            rumps.alert("No Profile Found", message="No voice profile is currently saved.")

    @rumps.clicked("Quit")
    def quit_app(self, _):
        self._listener.stop()
        rumps.quit_application()
