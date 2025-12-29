import pandas as pd
import numpy as np


def exponential_smoothing(series, alpha=0.3):
    """
    Apply exponential smoothing to time series data.
    
    Args:
        series: pandas Series with values to smooth
        alpha: Smoothing factor (0 < alpha < 1). Lower alpha = more smoothing.
    
    Returns:
        Smoothed pandas Series
    """
    smoothed = series.copy()
    smoothed.iloc[0] = series.iloc[0]
    
    for i in range(1, len(series)):
        smoothed.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * smoothed.iloc[i-1]
    
    return smoothed


def moving_average_smoothing(series, window=7):
    """
    Apply moving average smoothing to time series data.
    
    Args:
        series: pandas Series with values to smooth
        window: Size of moving average window
    
    Returns:
        Smoothed pandas Series
    """
    return series.rolling(window=window, center=True, min_periods=1).mean()


def smooth_data(df, method='exponential', alpha=0.3, window=7):
    """
    Apply smoothing to time series data.
    
    Args:
        df: DataFrame with 'unique_id', 'ds', 'y' columns
        method: 'exponential' or 'moving_average'
        alpha: Smoothing factor for exponential smoothing (0.3 = moderate smoothing)
        window: Window size for moving average (7 = weekly smoothing)
    
    Returns:
        DataFrame with smoothed 'y' values
    """
    df_smoothed = df.copy()
    
    if method == 'exponential':
        for uid in df['unique_id'].unique():
            mask = df['unique_id'] == uid
            df_smoothed.loc[mask, 'y'] = exponential_smoothing(df.loc[mask, 'y'], alpha=alpha).values
    elif method == 'moving_average':
        for uid in df['unique_id'].unique():
            mask = df['unique_id'] == uid
            df_smoothed.loc[mask, 'y'] = moving_average_smoothing(df.loc[mask, 'y'], window=window).values
    else:
        raise ValueError(f"Unknown smoothing method: {method}. Use 'exponential' or 'moving_average'")
    
    return df_smoothed


def split_train_test(df, test_start_date='2024-01-01'):
    df_train = df[df['ds'] < test_start_date].copy()
    df_test = df[df['ds'] >= test_start_date].copy()
    
    print(f"Training data: {df_train['ds'].min()} to {df_train['ds'].max()}")
    print(f"Testing data: {df_test['ds'].min()} to {df_test['ds'].max()}")
    
    return df_train, df_test


def merge_forecast_with_actual(df_test, forecasts):
    forecasts_reset = forecasts.reset_index()
    comparison = df_test.merge(forecasts_reset[['ds', 'NHITS']], on='ds')
    return comparison

