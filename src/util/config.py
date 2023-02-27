
"""
Config file

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import logging


class CONFIG:

    # Severity level of logging
    LOGS_LEVEL_DEEP_SPLIT = logging.INFO
    LOGS_LEVEL_VERIFIER = logging.CRITICAL  # logging.INFO
    LOGS_TERMINAL = True
    LOG_TO_FILE = False


logging.basicConfig(level=CONFIG.LOGS_LEVEL_DEEP_SPLIT)
