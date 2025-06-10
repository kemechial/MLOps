import os, sys, pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml


logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config: str):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]
        
        os.makedirs(RAW_DIR, exist_ok=True)
    
        logger.info(f"Data Ingestion started {self.bucket_name} and {self.file_name}")
    
    
    def download_csv_from_gcp(self):
        """
        Download CSV file from GCP bucket
        """
        try:
            client = storage.Client.from_service_account_json("neat-encoder-462512-g1-17effdd33d68.json")
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.download_to_filename(os.path.join(RAW_DIR, self.file_name))
            logger.info(f"File {self.file_name} downloaded from GCP bucket {self.bucket_name}")
        except Exception as e:
            raise CustomException("Failed to download file from GCP bucket", sys)
    
    def split_data(self):
        """
        Split the data into train and test sets
        """
        try:
            logger.info("Splitting data into train and test sets")
            df = pd.read_csv(os.path.join(RAW_DIR, self.file_name))
            train_df, test_df = train_test_split(df, test_size=1-self.train_test_ratio, random_state=42)
            df.to_csv(os.path.join(RAW_FILE))
            train_df.to_csv(os.path.join(TRAIN_FILE))
            test_df.to_csv(os.path.join(TEST_FILE))
            
            logger.info(f"Train data saved to {TRAIN_FILE}")
            logger.info(f"Test data saved to {TEST_FILE}")
            
        except Exception as e:
            raise CustomException("Failed to split data", e)
        
    def run(self):
        """
        Run the data ingestion pipeline
        """
        try:
            self.download_csv_from_gcp()
            self.split_data()
            logger.info("Data ingestion completed successfully")
        except CustomException as ce:
            logger.error(f"Data ingestion failed: {str(ce)}")
            raise CustomException("Data ingestion failed", sys)
        
        finally:
            logger.info("Data ingestion process finished")
            
            
            
if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
           