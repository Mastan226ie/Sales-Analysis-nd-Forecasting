import os
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from src.logger import logging
from src.exception import CustomException
import joblib
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class ForecastPipeline:
    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path

    def forecast(self, periods=6):
        try:
            # Load model
            with open(self.model_path, 'rb') as f:
                model = joblib.load(f)

            # Load past sales data
            df = pd.read_csv(self.data_path, index_col='Order Date', parse_dates=True)
            sales_series = df['Sales']

            # Generate forecast
            forecast = model.predict(n_periods=periods)

            # Generate future monthly dates
            forecast_dates = pd.date_range(sales_series.index[-1] + pd.DateOffset(months=1), periods=periods, freq='M')
            forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_dates)

            # Plot actual and forecasted data
            plt.figure(figsize=(12, 6))
            plt.plot(sales_series, label='Actual Sales')
            plt.plot(forecast_df['Forecast'], label='Forecasted Sales', color='orange')
            plt.title("Sales Forecast for Next 6 Months")
            plt.xlabel("Date")
            plt.ylabel("Sales")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Save plot and forecast data
            os.makedirs("artifacts", exist_ok=True)
            plt.savefig("artifacts/sales_forecast_plot.png")
            forecast_df.to_csv("artifacts/sales_forecast.csv")

            logging.info("Forecast plot and CSV saved successfully.")
            print("Forecast generated and saved successfully.")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    forecast = ForecastPipeline(
        model_path="artifacts/arima_model.pkl",
        data_path="artifacts/monthly_sales.csv"
    )
    forecast.forecast()
