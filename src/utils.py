# src/utils.py

import os
import dill
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.logger import logger

def save_object(file_path, obj):
    """
    Save Python object to a file using dill.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logger.info(f"Object saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving object: {e}")
        raise

def load_object(file_path):
    """
    Load Python object from a file using dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logger.error(f"Error loading object: {e}")
        raise

def evaluate_model(true, predicted):
    """
    Returns MAE, MSE, RMSE for model evaluation.
    """
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)

    logger.info(f"Evaluation -> MAE: {mae}, MSE: {mse}, RMSE: {rmse}")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}
