"""Constants and configuration for WhisperType."""

import os

SAMPLERATE = 16000
CHANNELS = 1
REALTIME = False
STREAM_STEP_MS = 400

CHUNK_SECONDS = 30
CHUNK_SAMPLES = CHUNK_SECONDS * SAMPLERATE

MODELS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-turbo-mlx",
}
DEFAULT_MODEL = "small"

# Audio thresholds
MIN_SPEECH_RMS = 0.005
TARGET_RMS = 0.1
NO_SPEECH_THRESHOLD = 0.8
LOG_PROB_THRESHOLD = -2.0
PRE_EMPHASIS = 0.97
WHISPER_PROMPT = "Voice dictation. Clear speech in English."

# LLM post-processing
LLM_ENABLED = False
LLM_MODEL = "qwen3:4b"
LLM_SYSTEM_PROMPT = (
    "You are a transcription editor. Clean up the raw speech-to-text output below:\n"
    "- Fix grammar, punctuation, and capitalization\n"
    "- Remove filler words (um, uh, like, you know, so, right)\n"
    "- Fix obvious transcription errors using context\n"
    "- Keep the original meaning and intent intact\n"
    "Output ONLY the cleaned text. No explanation, no quotes, nothing else."
)

# Speaker verification
VOICE_PROFILE_PATH = os.path.expanduser("~/.whisper/voice_profile.npy")
SPEAKER_SIM_THRESHOLD = 0.75
ENROLL_MIN_SECONDS = 3.0
