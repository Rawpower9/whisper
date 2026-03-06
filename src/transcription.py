"""Whisper transcription."""

import logging
import re

import mlx_whisper
import numpy as np

from .config import (
    NO_SPEECH_THRESHOLD,
    LOG_PROB_THRESHOLD,
    WHISPER_PROMPT,
)

log = logging.getLogger(__name__)


def transcribe(audio: np.ndarray, model_repo: str, *, partial: bool = False) -> str:
    """Run mlx-whisper on audio and return cleaned text.

    Args:
        audio: Float32 audio array (already prepared/normalised).
        model_repo: HuggingFace repo for the mlx-whisper model.
        partial: If True, use conservative temperature (single pass).
    """
    temperature = (0,) if partial else (0, 0.2, 0.4)
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=model_repo,
        temperature=temperature,
        condition_on_previous_text=False if partial else True,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        logprob_threshold=LOG_PROB_THRESHOLD,
        initial_prompt=WHISPER_PROMPT,
        verbose=False,
        language="en",
    )
    text = result["text"].strip()
    text = re.sub(r"\[.*?\]|\(.*?\)", "", text).strip()
    return text
