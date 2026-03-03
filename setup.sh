#!/bin/bash
set -e
echo "==> Installing whisper-cpp via Homebrew..."
brew install whisper-cpp sox
SIZE="large-v3"
echo "==> Downloading ggml-small model..."
mkdir -p ~/.whisper/models
SCRIPT="$(brew --prefix whisper-cpp)/models/download-ggml-model.sh"
if [ -f "$SCRIPT" ]; then
  bash "$SCRIPT" "$SIZE" ~/.whisper/models
else
  # Fallback: direct HuggingFace download
  curl -L \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$SIZE.bin" \
    -o ~/.whisper/models/ggml-$SIZE.bin
fi

echo "==> Installing Python dependencies..."
pip3 install -r requirements.txt

echo ""
echo "Setup complete. Run: python3 whisper_type.py"
echo ""
echo "IMPORTANT: Grant these macOS permissions to Terminal (or your terminal app):"
echo "  System Settings → Privacy & Security → Input Monitoring"
echo "  System Settings → Privacy & Security → Accessibility"
