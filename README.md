# WhisperType

A macOS menu bar app that transcribes your speech in real time. Hold **Right Option**, speak, release — and the transcribed text is typed wherever your cursor is.

Uses [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) for fast on-device transcription on Apple Silicon via Metal GPU.

## Features

- **Push-to-talk**: Hold Right Option to record, release to transcribe and type
- **On-device**: All processing happens locally — no cloud APIs
- **Metal GPU accelerated**: Uses MLX for fast inference on Apple Silicon
- **Floating overlay**: Shows transcription status while recording
- **Speaker verification** (optional): Enroll your voice to reject other speakers
- **LLM post-processing** (optional): Clean up transcriptions with a local Ollama model

## Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- [SoX](https://sox.sourceforge.net/) (`rec` command for audio capture)

## Setup

```bash
# Install SoX for audio recording
brew install sox

# Clone and set up
git clone <repo-url> && cd whisper
uv sync

# Run
uv run python whisper_type.py
```

### macOS Permissions

On first launch, grant these in **System Settings > Privacy & Security**:

1. **Input Monitoring** — for global hotkey detection (Right Option)
2. **Accessibility** — for typing transcribed text into other apps
3. **Microphone** — for audio recording

The app that needs permissions is your terminal (e.g. Terminal.app, iTerm2, Ghostty) when running from the command line.

## Usage

1. Launch the app — a **◉** icon appears in your menu bar
2. Place your cursor where you want text typed
3. **Hold Right Option** and speak (icon flashes **⏺**)
4. **Release Right Option** — text is transcribed and typed at your cursor

### Menu Bar Options

- **Enroll My Voice** — Record a voice sample for speaker verification
- **Clear Voice Profile** — Remove your voice profile and disable speaker verification
- **Quit** — Stop the app

## Project Structure

```
├── whisper_type.py        # Entry point
├── src/
│   ├── config.py          # Constants and configuration
│   ├── audio.py           # Audio preparation and speaker verification
│   ├── transcription.py   # Whisper transcription and LLM refinement
│   ├── overlay.py         # Floating overlay panel UI
│   └── app.py             # Main rumps menu bar application
├── build_app.sh           # Build standalone .app bundle with py2app
├── pyproject.toml
└── requirements.txt
```

## Configuration

Edit `src/config.py` to change:

| Setting | Default | Description |
|---------|---------|-------------|
| `DEFAULT_MODEL` | `"small"` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`, `turbo`) |
| `REALTIME` | `False` | Show partial transcriptions while recording |
| `LLM_ENABLED` | `False` | Post-process with local Ollama LLM |
| `LLM_MODEL` | `"qwen3:4b"` | Ollama model for post-processing |
| `SPEAKER_SIM_THRESHOLD` | `0.75` | Cosine similarity threshold for speaker verification |

## Building a Standalone App

```bash
bash build_app.sh
# Output: dist/WhisperType.app
cp -r dist/WhisperType.app /Applications/
```
