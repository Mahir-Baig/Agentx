from azure.storage.blob import BlobServiceClient
import argparse
from src.logger import logger
from src.exceptions import RagException
from dotenv import load_dotenv  
from src.config.constants import *
import sys

load_dotenv() 
azure_storageaccount_cs=  os.getenv("azure_storageaccount_cs")

class AzureBlobManager:
    def __init__(self, connection_string=azure_storageaccount_cs):
        logger.info("Initializing AzureBlobManager")
        self.storage_account_client = BlobServiceClient.from_connection_string(connection_string)
        

    def container_exists(self, container_name):
        logger.info(f"Checking if container {container_name} exists")
        try:
            container_client = self.storage_account_client.get_container_client(container_name)
            exists = container_client.exists()
            logger.info(f"Container {container_name} exists: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Failed to check if container {container_name} exists: {e}")
            raise RagException(f"Failed to check if container {container_name} exists", sys)

    def get_metadata(self, container_name, blob_name):
        logger.info(f"Getting metadata for {blob_name} in {container_name}")
        # if not self.container_exists(container_name):
        #     logger.error(f"Container {container_name} does not exist")
        #     raise RagException(f"Container {container_name} does not exist")
        try:
            container_client = self.storage_account_client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)
            properties = blob_client.get_blob_properties()
            metadata = {
                "container": container_name,
                "name": blob_name,
                "size": properties.size,
                "last_modified": properties.last_modified,
                "content_type": properties.content_settings.content_type
            }
            # logger.info(f"Metadata for {blob_name}: {metadata}")
            return metadata
        except Exception as e:
            logger.error(f"Failed to get metadata for {blob_name} in {container_name}: {e}")
            raise RagException(f"Failed to get metadata for {blob_name} in {container_name}", sys) 
        
    # @log_execution_time
    def list_blob_names_and_files(self, container_name):
        if not self.container_exists(container_name):
            logger.error(f"Container {container_name} does not exist")
            raise RagException(f"Container {container_name} does not exist")
        logger.info(f"Listing all blob names in {container_name}")
        try:
            container_client = self.storage_account_client.get_container_client(container_name)
            logger.info(f"Checking for blobs in container:{container_name}")
            blobs = container_client.list_blobs()
            blob_with_filenames = [blob.name for blob in blobs]
            blob_names = set([blob.split('/')[0] for blob in blob_with_filenames])
            logger.info(f"Blob names in {container_name}: {blob_names}")
            logger.info(f"Found {len(blob_names)} no of blobs in container:{container_name} ")
            
            # Log the number of folders in each blob
            for blob_name in blob_names:
                folder_count = sum(1 for blob in blob_with_filenames if blob.startswith(blob_name + '/'))
                logger.info(f"Blob {blob_name} contains {folder_count} folders")
            
            return blob_names, blob_with_filenames
        except Exception as e:
            logger.error(f"Failed to list blobs in {container_name}: {e}")
            raise RagException(f"Failed to list blobs in {container_name}", sys)
        
    # @log_execution_time
    def download_allfiles_in_container(self, container_name, download_dir):
        logger.info(f"Starting download of all files from {container_name} to {download_dir}")
        if not self.container_exists(container_name):
            logger.error(f"Container {container_name} does not exist")
            raise RagException(f"Container {container_name} does not exist", sys)
        try:
            container_client = self.storage_account_client.get_container_client(container_name)
            blobs = container_client.list_blobs()
            blob_count = 0 
            blob_list= []
            for blob in blobs:
                blob_count += 1
                blob_name = blob.name
                blob_list.append(blob_name)
                logger.info(f"Blob:{blob_name} is ready to get downloaded")
                download_path = os.path.join(download_dir, blob_name)
                os.makedirs(os.path.dirname(download_path), exist_ok=True)
                logger.info(f"Succesfully created {os.path.dirname(download_path)} directory to store the blob:{blob_name} files")
                
                # Download the blob directly
                blob_client = container_client.get_blob_client(blob_name)
                with open(download_path, "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())
        
                logger.info(f"Downloaded {blob_name} to {download_path}")
                # Log metadata
                metadata = self.get_metadata(container_name, blob_name)
                logger.info(f"Metadata for {blob_name}: {metadata}")
            
            logger.info(f"Total number of files found in {container_name}: {len(set(blob_list))} and they are -Blob Names:{set(blob_list)}")  # Log total blob count            
        except Exception as e:
            logger.error(f"Failed to download multiple files from {container_name}: {e}")
            raise RagException(f"Failed to download multiple files from {container_name}", sys)
        
    # @log_execution_time
    def download_allfiles_in_blob(self, container_name, download_dir, blob_name, existing_files):
        if not self.container_exists(container_name):
            logger.error(f"Container {container_name} does not exist")
            raise RagException(f"Container {container_name} does not exist", sys)
        
        logger.info(f"Starting download of files from {container_name}/{blob_name} to {download_dir}")
        try:
            container_client = self.storage_account_client.get_container_client(container_name)
            blobs = container_client.list_blobs(name_starts_with=blob_name)
            blob_with_filenames = []
            
            download_count = 0  # Counter for downloaded files
            
            for blob in blobs:
                if blob.name in existing_files:
                    logger.info(f"Skipping download of {blob.name} as it is already present in metadata and data ingestion data folder")
                    print(f"Skipping download of {blob.name} as it is already present in metadata and data ingestion data folder")
                    continue
                
                blob_client = container_client.get_blob_client(blob.name)
                blob_with_filenames.append(blob.name)
                blob_download_path = os.path.join(download_dir, blob.name)
                os.makedirs(os.path.dirname(blob_download_path), exist_ok=True)
                logger.info(f"Created directory for blob: {os.path.dirname(blob_download_path)}")
                
                with open(blob_download_path, "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())
                logger.info(f"Downloaded {blob.name} to {blob_download_path}")
                
                download_count += 1  # Increment counter for each downloaded file
            
            logger.info(f"Total number of files downloaded: {download_count}")
            download_path = os.path.join(download_dir,blob_name)
            logger.info(f"Downloaded path: {download_path}")
            return download_path, blob_with_filenames
        except Exception as e:
            logger.error(f"Failed to download multiple files from {container_name}/{blob_name}: {e}")
            raise RagException(f"Failed to download multiple files from {container_name}/{blob_name}", sys)

    # # ... ex
    # @log_execution_time
    # def download_allfiles_in_blob(self, container_name, download_dir, blob_name):
    #     if not self.container_exists(container_name):
    #         logger.error(f"Container {container_name} does not exist")
    #         raise RagException(f"Container {container_name} does not exist", sys)
        
    #     logger.info(f"Starting download of files from {container_name}/{blob_name} to {download_dir}")
    #     try:
    #         # os.makedirs(download_dir, exist_ok=True)
    #         # logger.info(f"Created download directory: {download_dir}")
            
    #         container_client = self.storage_account_client.get_container_client(container_name)
    #         blobs = container_client.list_blobs(name_starts_with=blob_name)
    #         blob_with_filenames = []
            
    #         download_count = 0  # Counter for downloaded files
            
    #         for blob in blobs:
    #             blob_client = container_client.get_blob_client(blob.name)
    #             blob_with_filenames.append(blob.name)
    #             blob_download_path = os.path.join(download_dir,blob.name)
    #             os.makedirs(os.path.dirname(blob_download_path), exist_ok=True)
    #             logger.info(f"Created directory for blob: {os.path.dirname(blob_download_path)}")
                
    #             with open(blob_download_path, "wb") as download_file:
    #                 download_file.write(blob_client.download_blob().readall())
    #             logger.info(f"Downloaded {blob.name} to {blob_download_path}")
                
    #             download_count += 1  # Increment counter for each downloaded file
            
    #         logger.info(f"Total number of files downloaded: {download_count}")
    #         download_path = os.path.dirname(blob_download_path)
    #         logger.info(f"downloaded path :{download_path}")
    #         return download_path,blob_with_filenames
    #     except Exception as e:
    #         logger.error(f"Failed to download multiple files from {container_name}/{blob_name}: {e}")
    #         raise RagException(f"Failed to download multiple files from {container_name}/{blob_name}", sys)

    # @log_execution_time
    def upload_files(self, container_name, folder_path, blob_name):
        logger.info(f"Starting upload of files from {folder_path} to {container_name}/{blob_name}")
        if not self.container_exists(container_name):
            logger.error(f"Container {container_name} does not exist")
            raise RagException(f"Container {container_name} does not exist", sys)
        try:
            container_client = self.storage_account_client.get_container_client(container_name)
            uploaded_count = 0  
            logger.info(f"Starting to upload all the files present in {folder_path}")
            logger.info(f"Total no of files present in folder:{folder_path} are {len(os.listdir(folder_path))}")
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    blob_path = os.path.join(blob_name, os.path.relpath(file_path, folder_path))
                    logger.info(f"Uploading {file_path} to {blob_path}")
                    blob_client = container_client.get_blob_client(blob_path)
                    with open(file_path, "rb") as data:
                        blob_client.upload_blob(data)
                    logger.info(f"Uploaded {file_path} to {blob_path}")
                    uploaded_count += 1  # Increment counter
                    # Log metadata
                    metadata = self.get_metadata(container_name, blob_path)
                    logger.info(f"Metadata for {blob_path}: {metadata}")
            logger.info(f"Total number of files uploaded: {uploaded_count}")  # Log total count
        except Exception as e:
            logger.error(f"Failed to upload files from {folder_path} to {container_name}/{blob_name}: {e}")
            raise RagException(e)

   

    # @log_execution_time
    def delete_files(self, container_name, blob_names):
        logger.info(f"Starting deletion of files from {container_name}")
        if not self.container_exists(container_name):
            logger.error(f"Container {container_name} does not exist")
            raise RagException(f"Container {container_name} does not exist", sys)
        try:
            container_client = self.storage_account_client.get_container_client(container_name)
            for blob_name in blob_names:
                logger.info(f"Deleting {blob_name}")
                blob_client = container_client.get_blob_client(blob_name)
                blob_client.delete_blob()
                logger.info(f"Deleted {blob_name}")
                # Log metadata
                metadata = self.get_metadata(container_name, blob_name)
                logger.info(f"Metadata for {blob_name}: {metadata}")
        except Exception as e:
            logger.error(f"Failed to delete files from {container_name}: {e}")
            raise RagException(e) 

def main():
    parser = argparse.ArgumentParser(description="Azure Blob Storage Manager")
    parser.add_argument("-c", "--connection_string", default=azure_storageaccount_cs, help="Azure Storage connection string")
    parser.add_argument("-dac", "--download_allfiles_in_container", nargs=2, metavar=("container_name", "download_dir"), help="Download all files from a container")
    parser.add_argument("-d", "--download_allfiles_in_blob", nargs=3, metavar=("container_name", "download_dir", "blob_name"), help="Download all files from a specific blob in a container ")
    parser.add_argument("-u", "--upload_files", nargs=3, metavar=("container_name", "folder_path", "blob_name"), help="Upload files to a container")
    parser.add_argument("-ls", "--list_blob_names_and_files", metavar="container_name", help="List all blob names and files in a container")
    parser.add_argument("-del", "--delete_files", nargs=2, metavar=("container_name", "blob_names"), help="Delete files from a container")

    args = parser.parse_args()

    try:
        functionalities = AzureBlobManager(args.connection_string)    
        logger.info("Blob storage connection is successful")
    except Exception as e:
        logger.error(f"Failed to connect to Azure Blob Storage: {e}")
        raise RagException("Failed to connect to Azure Blob Storage") from e

    if args.download_allfiles_in_container:
        container_name, download_dir = args.download_allfiles_in_container
        functionalities.download_allfiles_in_container(container_name, download_dir)
    elif args.download_allfiles_in_blob:
        container_name, download_dir, blob_name = args.download_allfiles_in_blob
        functionalities.download_allfiles_in_blob(container_name, download_dir, blob_name)
    elif args.upload_files:
        container_name, folder_path, blob_name = args.upload_files
        functionalities.upload_files(container_name, folder_path, blob_name)
    elif args.list_blob_names_and_files:
        container_name = args.list_blob_names_and_files
        blob_names, blob_with_filenames = functionalities.list_blob_names_and_files(container_name)
        print(f"Blob names: {blob_names}")
        print(f"Blob names with files: {blob_with_filenames}")
    elif args.delete_files:
        container_name, blob_names = args.delete_files
        blob_names = blob_names.split(",")  # Assuming blob names are comma-separated
        functionalities.delete_files(container_name, blob_names)

if __name__ == "__main__":
    main()


