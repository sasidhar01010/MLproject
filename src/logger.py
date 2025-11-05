import logging
import os
from datetime import datetime
import sys 

# --- Setup Log Directory and File Path ---

logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Create log file path
LOG_FILE = f"app_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# --- Define Logger Configuration ---

# 1. Get a specific logger instance
logger = logging.getLogger("ML_PROJECT_LOGGER") 
logger.setLevel(logging.INFO) # Set the lowest level to handle

# 2. Define the log message format
formatter = logging.Formatter(
    "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

# 3. Create Handlers
# File Handler (writes to the log file)
file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a')
file_handler.setFormatter(formatter)

# Stream Handler (writes to the console for real-time feedback)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

# 4. Add Handlers to the Logger
# Only add handlers if the logger doesn't have any yet (prevents duplication on multiple imports)
if not logger.handlers: 
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)