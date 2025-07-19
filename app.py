import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from src.utils import load_object

# App title
st.title("Sales Forecasting App using ARIMA")
st.markdown("Predict future monthly sales using a trained ARIMA model.")

# File paths
MODEL_PATH = "artifacts/arima_model.pkl"
DATA_PATH = "artifacts/monthly_sales.csv"

# Load the model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Load the data
try:
    df = pd.read_csv(DATA_PATH, parse_dates=["Order Date"], index_col="Order Date")
    sales_series = df['Sales']
except Exception as e:
    st.error(f"Failed to load sales data: {e}")
    st.stop()

# Select months to forecast
n_periods = st.slider("Select number of future months to forecast", min_value=1, max_value=24, value=6)

# Forecast using pmdarima
try:
    forecast_values = model.predict(n_periods=n_periods)
except Exception as e:
    st.error(f"Forecasting failed: {e}")
    st.stop()

# Build forecast index
forecast_index = pd.date_range(start=sales_series.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='M')
forecast_series = pd.Series(forecast_values, index=forecast_index)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
sales_series[-12:].plot(ax=ax, label="Historical Sales", color="blue")
forecast_series.plot(ax=ax, label="Forecast", color="red")
ax.set_title("Sales Forecast for Next {} Months".format(n_periods))
ax.set_ylabel("Sales")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Display forecast table
st.subheader("Forecasted Sales Values")
forecast_df = pd.DataFrame({
    "Date": forecast_index,
    "Predicted Sales": forecast_values
})
st.dataframe(forecast_df.reset_index(drop=True))
