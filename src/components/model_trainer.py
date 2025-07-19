# src/components/model_trainer.py

import os
import sys
import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "arima_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def train_model(self, transformed_data_path):
        try:
            df = pd.read_csv(transformed_data_path, parse_dates=['Order Date'], index_col='Order Date')
            
            # Train-test split (80% train, 20% test)
            size = int(len(df) * 0.8)
            train, test = df.iloc[:size], df.iloc[size:]

            # Fit ARIMA
            model = ARIMA(train, order=(1, 1, 1))  # default order; should be tuned
            model_fit = model.fit()

            # Forecast
            forecast = model_fit.forecast(steps=len(test))
            mae = mean_absolute_error(test, forecast)

            logging.info(f"ARIMA model trained. MAE: {mae}")
            print(f"Model MAE: {mae:.2f}")

            # Save the model
            os.makedirs(os.path.dirname(self.model_trainer_config.model_path), exist_ok=True)
            joblib.dump(model_fit, self.model_trainer_config.model_path)

            logging.info("ARIMA model saved successfully.")
            return self.model_trainer_config.model_path

        except Exception as e:
            raise CustomException(e, sys)
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_model(transformed_data_path="artifacts/monthly_sales.csv")
