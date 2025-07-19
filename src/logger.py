# src/logger.py

import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file name with timestamp
LOG_FILE = f"{LOG_DIR}/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Set up basic logging
logging.basicConfig(
    filename=LOG_FILE,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    level=logging.INFO,
    filemode="w"
)

# Shortcut to log
logger = logging.getLogger()
