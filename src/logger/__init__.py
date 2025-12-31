import logging
import os
from src.config.constants import TIMESTAMP
import shutil
import socket

LOG_DIR = "logs"

class HostnameFilter(logging.Filter):
    """Custom filter to add hostname to log records"""
    def filter(self, record):
        record.user = socket.gethostname()
        return True

def get_log_file_name():
    return f"log_{TIMESTAMP}.log"


LOG_FILE_NAME = get_log_file_name()

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

logging.basicConfig(filename=LOG_FILE_PATH,
                    filemode="w",
                    format="[%(asctime)s]: %(levelname)s -['User_Name']:mahir -['filepath']:%(pathname)s ['filename']:%(filename)s -['function_name']:%(funcName)s -['line_no']:%(lineno)d - %(message)s",
                    level=logging.INFO
                    )

logger = logging.getLogger("RagAgent")
# logger.addFilter(HostnameFilter())
# Clean up old log files (but don't delete the directory itself)
try:
    for filename in os.listdir(LOG_DIR):
        if filename.startswith("log_") and filename != LOG_FILE_NAME:
            file_path = os.path.join(LOG_DIR, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass  # Skip if file is locked
except Exception:
    pass  # If anything fails, just continue