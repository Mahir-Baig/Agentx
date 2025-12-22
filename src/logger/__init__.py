import logging
import os
from src.config.constants import TIMESTAMP
import shutil
import socket
import sys

# Force UTF-8 encoding for Windows console - must be done EARLY
if sys.platform == "win32":
    # Set environment variable for Python to use UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # Reconfigure standard streams
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except Exception:
        try:
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass  # If all fails, continue

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

# Create handlers with UTF-8 encoding
file_handler = logging.FileHandler(LOG_FILE_PATH, mode="w", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("[%(asctime)s]: %(levelname)s - %(message)s"))

# Console handler with UTF-8 encoding and error handling
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("[%(asctime)s]: %(levelname)s - %(message)s"))

# Add a filter to handle encoding errors gracefully
class SafeEncodingFilter(logging.Filter):
    def filter(self, record):
        # Ensure the message can be safely encoded/displayed
        if isinstance(record.msg, str):
            try:
                # Try to encode as UTF-8
                record.msg.encode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                # If it fails, strip emojis and non-ASCII characters
                record.msg = record.msg.encode('ascii', errors='ignore').decode('ascii')
        # Also handle formatted message
        if hasattr(record, 'message') and isinstance(record.message, str):
            try:
                record.message.encode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                record.message = record.message.encode('ascii', errors='ignore').decode('ascii')
        return True

# Always apply filter to be safe
console_handler.addFilter(SafeEncodingFilter())

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger("RagAgent")
logger.setLevel(logging.INFO)
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