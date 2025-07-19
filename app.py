import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMAResults
from src.utils import load_object

# App title
st.title("Sales Forecasting App using ARIMA")
st.markdown("Predict future monthly sales using a trained ARIMA model.")

# Load the model
MODEL_PATH = "artifacts/arima_model.pkl"
DATA_PATH = "artifacts/monthly_sales.csv"

try:
    model: ARIMAResults = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Load the monthly sales data
try:
    df = pd.read_csv(DATA_PATH, parse_dates=["Order Date"], index_col="Order Date")
except Exception as e:
    st.error(f"Failed to load sales data: {e}")
    st.stop()

# Slider for number of months to forecast
n_periods = st.slider("Select number of future months to forecast", min_value=1, max_value=24, value=6)

# Perform forecasting
forecast = model.forecast(steps=n_periods)

# Plot the results
fig, ax = plt.subplots(figsize=(10, 5))
df[-12:].plot(ax=ax, label="Historical Sales")
forecast.index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='M')
forecast.plot(ax=ax, label="Forecast", color="red")
ax.set_title("Sales Forecast")
ax.set_ylabel("Sales")
ax.legend()

# Show plot
st.pyplot(fig)

# Show forecast table
st.subheader("Forecasted Sales Values")
forecast_df = pd.DataFrame({
    "Date": forecast.index,
    "Predicted Sales": forecast.values
})
st.dataframe(forecast_df)