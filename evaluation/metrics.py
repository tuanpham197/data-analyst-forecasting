import numpy as np
import pandas as pd


def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mape(y_true, y_pred):
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_forecast(df_test, forecasts, model_name='NHITS'):
    forecasts_reset = forecasts.reset_index()
    comparison = df_test.merge(forecasts_reset[['ds', model_name]], on='ds', how='inner')
    
    y_true = comparison['y'].values
    y_pred = comparison[model_name].values
    
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }
    
    return metrics, comparison

