import os
import sys
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    """
    Data ingestion configuration
    """
    train_data_path: str = "data/train.csv"
    test_data_path: str = "data/test.csv"

class DataIngestion:
    """
    Handles data loading and basic validation
    """
    def __init__(self):
        self.config = DataIngestionConfig()
        logging.info("DataIngestion initialized")
    
    def initiate_data_ingestion(self):
        """
        Loading and return train and test data
        """
        logging.info("Data ingestion started")

        try:
            # Load train data
            train_df = pd.read_csv(self.config.train_data_path)
            logging.info(f"Train data loaded: {train_df.shape}")

            # Load test data
            test_df = pd.read_csv(self.config.test_data_path)
            logging.info(f"Test data loaded: {test_df.shape}")

            logging.info("Data ingesiton completed")

            return train_df, test_df
        
        except Exception as e:
            logging.info("Error in data ingestion")
            raise CustomException(e)
        
if __name__ == "__main__":
    # Test the component
    obj = DataIngestion()
    train, test = obj.initiate_data_ingestion()
    print(f"✅ Train shape: {train.shape}")
    print(f"✅ Test shape: {test.shape}")
