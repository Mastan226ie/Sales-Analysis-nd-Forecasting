import os
import pandas as pd
from src.exception import CustomException
from src.logger import logger
import sys

class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join("artifacts", "raw.csv")

    def initiate_data_ingestion(self):
        logger.info("Starting Data Ingestion process")

        try:
            # Read the dataset
            df = pd.read_csv("notebook/data/Sales.csv")
            logger.info(" Sales dataset read successfully")

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.raw_data_path, index=False)
            logger.info(f" Raw data saved at: {self.raw_data_path}")

            return self.raw_data_path

        except Exception as e:
            logger.error(" Data ingestion failed")
            raise CustomException(e, sys)
if __name__=="__main__":
    ingestor = DataIngestion()
    data_path = ingestor.initiate_data_ingestion()