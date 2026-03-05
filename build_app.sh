#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

APP_NAME="WhisperType"
ICON_FILE="icon.icns"

echo "=== Building $APP_NAME.app ==="

# Ensure uv is available
if ! command -v uv &>/dev/null; then
    echo "Error: uv is not installed. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install dependencies including py2app
uv pip install py2app --quiet 2>/dev/null || pip install py2app

# Build icon if iconutil is available and no .icns exists
if [ ! -f "$ICON_FILE" ]; then
    echo "No icon.icns found — app will use default Python icon."
    ICON_OPT=""
else
    ICON_OPT="'icon': '$ICON_FILE',"
fi

# Create a minimal setup.py for py2app
cat > setup_app.py << 'SETUP_EOF'
import os
from setuptools import setup

APP = ['whisper_type.py']
DATA_FILES = []

OPTIONS = {
    'argv_emulation': False,
    'plist': {
        'CFBundleName': 'WhisperType',
        'CFBundleDisplayName': 'WhisperType',
        'CFBundleIdentifier': 'com.whispertype.app',
        'CFBundleVersion': '0.1.0',
        'CFBundleShortVersionString': '0.1.0',
        'LSBackgroundOnly': False,
        'LSUIElement': True,  # Menu bar app — no Dock icon
        'NSMicrophoneUsageDescription': 'WhisperType needs microphone access to record and transcribe speech.',
        'NSAppleEventsUsageDescription': 'WhisperType needs accessibility access to type transcribed text.',
    },
    'packages': [
        'src',
        'rumps',
        'pynput',
        'numpy',
        'ollama',
        'resemblyzer',
        'mlx_whisper',
        'huggingface_hub',
    ],
    'includes': [],
    'frameworks': [],
    'resources': [],
}

# Add icon if it exists
if os.path.exists('icon.icns'):
    OPTIONS['iconfile'] = 'icon.icns'

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
SETUP_EOF

# Clean previous builds
rm -rf build dist

echo "Running py2app..."
python setup_app.py py2app 2>&1 | tail -5

if [ -d "dist/$APP_NAME.app" ]; then
    echo ""
    echo "=== Build successful! ==="
    echo "App location: $(pwd)/dist/$APP_NAME.app"
    echo ""
    echo "To install, run:"
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

# Clean up generated setup file
rm -f setup_app.py
