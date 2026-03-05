#!/usr/bin/env python3
"""WhisperType — Hold Right Option to record, release to transcribe & type."""

import logging

from src.config import DEFAULT_MODEL
from src.app import WhisperTypeApp

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

if __name__ == "__main__":
    log.info("Starting with model: %s", DEFAULT_MODEL)
    WhisperTypeApp(DEFAULT_MODEL).run()
