"""Audio preparation and speaker verification."""

import logging
import os
import threading

import numpy as np

from .config import (
    MIN_SPEECH_RMS,
    TARGET_RMS,
    PRE_EMPHASIS,
    SAMPLERATE,
    SPEAKER_SIM_THRESHOLD,
    VOICE_PROFILE_PATH,
)

log = logging.getLogger(__name__)

_voice_encoder: "object | None" = None
_voice_encoder_lock = threading.Lock()


def _get_voice_encoder():
    global _voice_encoder
    with _voice_encoder_lock:
        if _voice_encoder is None:
            try:
                from resemblyzer import VoiceEncoder
                _voice_encoder = VoiceEncoder()
                log.info("VoiceEncoder loaded")
            except Exception:
                log.exception("Failed to load VoiceEncoder; speaker verification disabled")
        return _voice_encoder


def speaker_similarity(audio: np.ndarray) -> "float | None":
    """Cosine similarity vs stored profile, or None if no profile / error."""
    if not os.path.isfile(VOICE_PROFILE_PATH):
        return None
    encoder = _get_voice_encoder()
    if encoder is None:
        return None
    try:
        from resemblyzer import preprocess_wav
        profile = np.load(VOICE_PROFILE_PATH)
        wav = preprocess_wav(audio, source_sr=SAMPLERATE)
        with _voice_encoder_lock:
            embedding = encoder.embed_utterance(wav)
        sim = float(np.dot(profile, embedding) /
                    (np.linalg.norm(profile) * np.linalg.norm(embedding) + 1e-9))
        log.debug("Speaker similarity: %.3f (threshold %.2f)", sim, SPEAKER_SIM_THRESHOLD)
        return sim
    except Exception:
        log.exception("Speaker similarity check failed; passing audio through")
        return None


def enroll_voice(audio: np.ndarray) -> "np.ndarray | None":
    """Compute and return a voice embedding, or None on failure."""
    encoder = _get_voice_encoder()
    if encoder is None:
        return None
    try:
        from resemblyzer import preprocess_wav
        wav = preprocess_wav(audio, source_sr=SAMPLERATE)
        with _voice_encoder_lock:
            return encoder.embed_utterance(wav)
    except Exception:
        log.exception("Voice enrollment embedding failed")
        return None


def prepare_audio(audio_int16: np.ndarray) -> "np.ndarray | None":
    """Convert int16 audio to float32, check energy, and normalise for Whisper.

    Returns None when the recording is too quiet to be speech.
    """
    audio = audio_int16.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(audio ** 2)))
    log.debug("Audio RMS: %.5f", rms)
    if rms < MIN_SPEECH_RMS:
        log.debug("Audio too quiet (RMS %.5f < %.5f); skipping whisper", rms, MIN_SPEECH_RMS)
        return None
    audio = audio * (TARGET_RMS / rms)
    audio = np.clip(audio, -1.0, 1.0)
    audio = np.append(audio[0], audio[1:] - PRE_EMPHASIS * audio[:-1])
    return audio
