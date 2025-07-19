import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    transformed_data_path: str = os.path.join("artifacts", "monthly_sales.csv")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def transform_data(self, raw_data_path):
        try:
            logging.info("Reading raw sales data...")
            df = pd.read_csv(raw_data_path)

            logging.info("Converting Order Date to datetime and resampling sales monthly...")
            df['Order Date'] = pd.to_datetime(df['Order Date'])
            df = df.sort_values('Order Date')
            df.set_index('Order Date', inplace=True)

            monthly_sales = df['Sales'].resample('M').sum()

            os.makedirs(os.path.dirname(self.transformation_config.transformed_data_path), exist_ok=True)
            monthly_sales.to_csv(self.transformation_config.transformed_data_path)

            logging.info(f"Monthly sales data saved to {self.transformation_config.transformed_data_path}")
            return self.transformation_config.transformed_data_path

        except Exception as e:
            raise CustomException(e, sys)


# For standalone execution
if __name__ == "__main__":
    try:
        obj = DataTransformation()
        raw_path = os.path.join("artifacts", "raw.csv")
        transformed_path = obj.transform_data(raw_path)
        print(f"Transformed data saved at: {transformed_path}")
    except Exception as e:
        print(f"Transformation failed: {e}")
