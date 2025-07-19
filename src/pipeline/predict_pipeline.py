import os
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from src.logger import logging
from src.exception import CustomException
import joblib

class ForecastPipeline:
    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path

    def forecast(self, periods=6):
        try:
            with open(self.model_path, 'rb') as f:
                model = joblib.load(f)

            df = pd.read_csv(self.data_path, index_col='Order Date', parse_dates=True)
            forecast = model.forecast(steps=periods)
            
            # Generate future dates
            forecast_dates = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=periods, freq='ME')
            forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_dates)

            # Plotting
            plt.figure(figsize=(12,6))
            plt.plot(df, label='Actual Sales')
            plt.plot(forecast_df, label='Forecasted Sales', color='orange')
            plt.title("Sales Forecast for Next 6 Months")
            plt.xlabel("Date")
            plt.ylabel("Sales")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            os.makedirs("artifacts", exist_ok=True)
            plt.savefig("artifacts/sales_forecast_plot.png")
            forecast_df.to_csv("artifacts/sales_forecast.csv")

            logging.info("Forecast plot and CSV saved successfully")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    forecast = ForecastPipeline(
        model_path="artifacts/arima_model.pkl",
        data_path="artifacts/monthly_sales.csv"
    )
    forecast.forecast()