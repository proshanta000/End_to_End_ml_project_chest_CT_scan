import os
import zipfile
import gdown
from chestCancerClassifier import logger
from chestCancerClassifier.utils.common import get_size
from chestCancerClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    """
    Handles the data ingestion process, including downloading and extracting data.
    """
    def __init__(self, config: DataIngestionConfig):
        # Initialize the class with a DataIngestionConfig object
        self.config = config
    
    def download_file(self) -> str:
        """
        Downloads a file from a Google Drive URL.
        
        This method uses the `gdown` library to download a file, which is specifically
        designed to handle Google Drive links. It extracts the file ID and
        constructs the appropriate URL for the download.
        
        Returns:
            str: The path to the downloaded zip file.
        """
        try:
            # Get the Google Drive URL from the configuration
            dataset_url = self.config.source_URL
            # Get the local path where the zip file will be saved
            Zip_download_dir = self.config.local_data_file
            
            # Create the artifacts directory if it doesn't exist
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {Zip_download_dir}")

            # Extract the file ID from the Google Drive URL
            file_id = dataset_url.split("/")[-2]
            # Construct the download prefix for gdown
            prefix = "https://drive.google.com/uc?/export=download&id="
            
            # Use gdown to download the file to the specified directory
            gdown.download(prefix + file_id, Zip_download_dir)

            logger.info(f"File downloaded to {Zip_download_dir}")
            return Zip_download_dir

        except Exception as e:
            # Log any exceptions that occur during the download process
            logger.error(f"Error downloading file: {e}")
            raise e
    
    def extract_zip_file(self):
        """
        Extracts a zip file into a specified directory.
        
        This method takes the path of a downloaded zip file and extracts its
        contents to the designated `unzip_dir`. It ensures the destination
        directory exists before extraction.
        """
        # Get the path where the zip file should be extracted
        unzip_path = self.config.unzip_dir
        
        # Create the extraction directory if it doesn't exist
        os.makedirs(unzip_path, exist_ok=True)
        
        # Open the zip file in read mode and extract all its contents
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        
        logger.info(f"Zip file extracted to {unzip_path}")