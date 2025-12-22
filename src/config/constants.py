from datetime import datetime
import socket
import os
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ARTIFACT_DIR = os.path.join("artifacts",TIMESTAMP)
USERNAME = socket.gethostname()
ERROR_COLLECTION_NAME="rag_error_logs"
ACCEPTED_CONTAINER_NAME="accepted"
REJECTED_CONTAINER_NAME="rejected"