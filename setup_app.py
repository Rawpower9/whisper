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
        'LSUIElement': True,
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

if os.path.exists('icon.icns'):
    OPTIONS['iconfile'] = 'icon.icns'

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
