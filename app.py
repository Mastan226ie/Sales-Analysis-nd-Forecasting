import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
#from src.utils import load_object
from io import BytesIO
import warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# App title
st.title("Sales Forecasting App using ARIMA")
st.markdown("Predict future monthly sales using a trained ARIMA model.")

# Paths
MODEL_PATH = "artifacts/arima_model.pkl"
DATA_PATH = "artifacts/monthly_sales.csv"
MAE_PATH = "artifacts/mae.txt"

# Load the model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Load the data
try:
    df = pd.read_csv(DATA_PATH, parse_dates=["Order Date"], index_col="Order Date")
except Exception as e:
    st.error(f"Failed to load sales data: {e}")
    st.stop()

# Display model MAE if available
try:
    with open(MAE_PATH, "r") as f:
        mae = float(f.read())
    st.success(f"Model Mean Absolute Error (MAE): {mae:.2f}")
except:
    st.warning("Model MAE not available.")

# Select months to forecast
n_periods = st.slider("Select number of future months to forecast", min_value=1, max_value=24, value=6)

# Forecast using pmdarima with confidence intervals
try:
    forecast_values, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
except Exception as e:
    st.error(f"Forecasting failed: {e}")
    st.stop()

# Build forecast index
forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='M')
forecast_series = pd.Series(forecast_values, index=forecast_index)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
df[-12:]["Sales"].plot(ax=ax, label="Historical Sales", color="blue")
forecast_series.plot(ax=ax, label="Forecast", color="red")
ax.fill_between(forecast_index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
ax.set_title("Sales Forecast")
ax.set_ylabel("Sales")
ax.legend()
st.pyplot(fig)

# Forecast DataFrame
forecast_df = pd.DataFrame({
    "Date": forecast_index,
    "Predicted Sales": forecast_values,
    "Lower Bound": conf_int[:, 0],
    "Upper Bound": conf_int[:, 1]
})
st.subheader("Forecasted Sales Table")
st.dataframe(forecast_df)

# CSV download button
csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Forecast as CSV",
    data=csv,
    file_name='sales_forecast.csv',
    mime='text/csv'
)
