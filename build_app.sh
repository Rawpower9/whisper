#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

APP_NAME="WhisperType"

echo "=== Building $APP_NAME.app ==="

# Ensure uv is available
if ! command -v uv &>/dev/null; then
    echo "Error: uv is not installed. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Sync project deps and install PyInstaller
uv sync --quiet
uv pip install --python .venv/bin/python3 pyinstaller --quiet

# Remove the obsolete 'typing' backport that conflicts with PyInstaller
# (pulled in by resemblyzer but not actually needed on Python 3.12+)
uv pip uninstall --python .venv/bin/python3 typing 2>/dev/null || true

# Clean previous builds
rm -rf build dist "${APP_NAME}.spec"

# Build the .app bundle
echo "Running PyInstaller..."
.venv/bin/python3 -m PyInstaller \
    --name "$APP_NAME" \
    --windowed \
    --noconfirm \
    --osx-bundle-identifier com.whispertype.app \
    ${ICON_FILE:+"--icon" "$ICON_FILE"} \
    --hidden-import=rumps \
    --hidden-import=pynput.keyboard._darwin \
    --hidden-import=AppKit \
    --hidden-import=PyObjCTools.AppHelper \
    --hidden-import=mlx.core \
    --hidden-import=mlx_whisper \
    --hidden-import=huggingface_hub \
    --hidden-import=resemblyzer \
    --collect-all mlx \
    --collect-all mlx_whisper \
    --collect-all resemblyzer \
    --collect-all rumps \
    --collect-submodules pynput \
    whisper_type.py

# Patch Info.plist to add required permission descriptions and LSUIElement
PLIST="dist/$APP_NAME.app/Contents/Info.plist"
if [ -f "$PLIST" ]; then
    # LSUIElement: menu bar app, no Dock icon
    /usr/libexec/PlistBuddy -c "Add :LSUIElement bool true" "$PLIST" 2>/dev/null || \
    /usr/libexec/PlistBuddy -c "Set :LSUIElement true" "$PLIST"

    # Permission descriptions shown in macOS prompts
    /usr/libexec/PlistBuddy -c "Add :NSMicrophoneUsageDescription string 'WhisperType needs microphone access to record and transcribe speech.'" "$PLIST" 2>/dev/null || true
    /usr/libexec/PlistBuddy -c "Add :NSAppleEventsUsageDescription string 'WhisperType needs accessibility access to type transcribed text.'" "$PLIST" 2>/dev/null || true
fi

if [ -d "dist/$APP_NAME.app" ]; then
    echo ""
    echo "=== Build successful! ==="
    echo "App location: $(pwd)/dist/$APP_NAME.app"
    echo ""
    echo "To install:"
    echo "  cp -r dist/$APP_NAME.app /Applications/"
    echo ""
    echo "To run now:"
    echo "  open dist/$APP_NAME.app"
    echo ""
    echo "Note: On first launch, macOS will ask for:"
    echo "  1. Microphone access"
    echo "  2. Accessibility access (for typing)"
    echo "  3. Input Monitoring (for key detection)"
    echo "Grant these in System Settings > Privacy & Security."
else
    echo "Build failed. Check output above for errors."
    exit 1
fi
