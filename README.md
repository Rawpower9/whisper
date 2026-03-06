# WhisperType

A macOS menu bar app that transcribes your speech in real time. Hold **Right Option**, speak, release — and the transcribed text is typed wherever your cursor is.

Uses [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) for fast on-device transcription on Apple Silicon via Metal GPU.

## Features

- **Push-to-talk**: Hold Right Option to record, release to transcribe and type
- **On-device**: All processing happens locally — no cloud APIs, no data leaves your machine
- **Metal GPU accelerated**: Uses MLX for fast inference on Apple Silicon
- **Floating overlay**: Shows transcription status while recording
- **Speaker verification** (optional): Enroll your voice to reject other speakers

## Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- [SoX](https://sox.sourceforge.net/) (`rec` command for audio capture)
- [Homebrew](https://brew.sh/) (for installing SoX)

## Setup

### 1. Install system dependencies

```bash
brew install sox
```

### 2. Clone and install

```bash
git clone <repo-url>
cd whisper
uv sync
```

This creates a `.venv` virtual environment and installs all Python dependencies (`mlx-whisper`, `rumps`, `pynput`, `numpy`, `resemblyzer`).

### 3. Grant macOS permissions

The app needs three permissions in **System Settings > Privacy & Security**. macOS will prompt you on first use, but you can grant them ahead of time.

The permissions apply to whichever app is running the process — your **terminal** (Terminal.app, iTerm2, Ghostty, etc.) when running from the command line, or **WhisperType.app** if using the standalone build.

| Permission | Why |
|---|---|
| **Input Monitoring** | Detect the Right Option key globally |
| **Accessibility** | Type transcribed text into other apps |
| **Microphone** | Record audio |

### 4. Run

```bash
uv run python whisper_type.py
```

A **◉** icon appears in your menu bar — you're ready to go.

## Usage

1. Place your cursor where you want text typed
2. **Hold Right Option** and speak (menu bar icon flashes **⏺**)
3. **Release Right Option** — text is transcribed and typed at your cursor

### Menu Bar Options

- **Enroll My Voice** — Record a voice sample for speaker verification (hold Right Option for 3+ seconds after clicking)
- **Clear Voice Profile** — Remove your voice profile and disable speaker verification
- **Quit** — Stop the app

## Configuration

Edit `src/config.py` to change:

| Setting | Default | Description |
|---|---|---|
| `DEFAULT_MODEL` | `"small"` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`, `turbo`) |
| `REALTIME` | `False` | Show partial transcriptions while recording |
| `SPEAKER_SIM_THRESHOLD` | `0.75` | Cosine similarity threshold for speaker verification |
| `MIN_SPEECH_RMS` | `0.005` | Minimum audio level to trigger transcription |

Models are downloaded automatically from HuggingFace on first use.

## Building a Standalone App

Build a self-contained `.app` bundle with PyInstaller:

```bash
bash build_app.sh
```

This produces `dist/WhisperType.app`. To install:

```bash
cp -r dist/WhisperType.app /Applications/
open /Applications/WhisperType.app
```

## Project Structure

```
├── whisper_type.py        # Entry point
├── src/
│   ├── config.py          # Constants and configuration
│   ├── audio.py           # Audio preparation and speaker verification
│   ├── transcription.py   # Whisper transcription via mlx-whisper
│   ├── overlay.py         # Floating overlay panel UI
│   └── app.py             # Main rumps menu bar application
├── build_app.sh           # Build standalone .app with PyInstaller
├── pyproject.toml
└── requirements.txt
```
