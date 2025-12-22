from src.services.azure_blob_service import AzureBlobManager
from src.logger import logger
from src.exceptions import RagException
import sys

def ingest_files_from_azure_blob(container_name: str, download_dir: str, blob_name: str = None):
    """
    Ingest files from Azure Blob Storage to a local directory.

    :param container_name: Name of the Azure Blob container.
    :param download_dir: Local directory to download files to.
    :param blob_name: Specific blob name to download files from (optional).
    """
    try:
        azure_blob_manager = AzureBlobManager()

        if blob_name:
            logger.info(f"Starting ingestion of files from blob '{blob_name}' in container '{container_name}' to '{download_dir}'")
            azure_blob_manager.download_allfiles_in_blob(container_name, download_dir, blob_name, existing_files=[])
            return download_dir
        else:
            logger.info(f"Starting ingestion of all files from container '{container_name}' to '{download_dir}'")
            azure_blob_manager.download_allfiles_in_container(container_name, download_dir)
            return download_dir
            logger.info(f"Completed ingestion of files from container '{container_name}' to '{download_dir}'")
    except Exception as e:
        logger.error(f"Error during ingestion from Azure Blob Storage: {e}")
        raise RagException(f"Error during ingestion from Azure Blob Storage: {e}", sys)
