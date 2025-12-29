import pandas as pd
import numpy as np


def generate_fashion_data():
    dates = pd.date_range(start='2023-01-01', end='2024-01-31', freq='D')
    n = len(dates)
    
    trend = np.linspace(100, 150, n)
    
    seasonality = 30 * np.sin(2 * np.pi * dates.dayofyear / 365) + \
                  20 * np.sin(4 * np.pi * dates.dayofyear / 365)
    
    noise = np.random.normal(0, 10, n)
    
    y = trend + seasonality + noise
    
    df = pd.DataFrame({
        'unique_id': 'fashion_store_1',
        'ds': dates,
        'y': y
    })
    return df

