import pandas as pd
import numpy as np

def prepare_retail_data():
    """
    Prepares the retail store inventory dataset for forecasting.
    Improvements made:
    1. Encoded weather conditions properly before outlier detection.
    2. Detected and handled outliers in 'Units Sold' using IQR method.
    3. Added feature engineering: cyclical features for day_of_week and month, weekend flag.
    4. Provided detailed train-test split information.
    """

    # ============================================================
    # 1. LOAD AND PREPARE DATA
    # ============================================================
    # Load the retail inventory dataset
    df_raw = pd.read_csv('./files/retail_store_inventory.csv')

    # Convert Date column to datetime
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])

    # Filter for August, September, October, November, December 2023 (5 months)
    df_raw = df_raw[
        (df_raw['Date'] >= '2023-08-01') &
        (df_raw['Date'] <= '2023-12-31')
    ].copy()

    # Create a unique identifier for each Store-Product combination
    df_raw['unique_id'] = df_raw['Store ID'] + '_' + df_raw['Product ID']

    # Prepare data in NeuralForecast format (long-panel)
    df = df_raw[['unique_id', 'Date', 'Units Sold', 'Price', 'Discount',
                'Holiday/Promotion', 'Weather Condition']].copy()
    df.columns = ['unique_id', 'ds', 'y', 'price', 'discount', 'holiday', 'weather']

    # Sort by unique_id and date
    df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

    # Display data info
    print("="*60)
    print("OVERALL DATASET INFO")
    print("="*60)
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print(f"Number of unique series (Store-Product): {df['unique_id'].nunique()}")
    print(f"Total days available: {(df['ds'].max() - df['ds'].min()).days + 1}")

    # ============================================================
    # IMPROVEMENT 1: ENCODE WEATHER PROPERLY (BEFORE OUTLIER DETECTION)
    # ============================================================
    print("\n" + "="*60)
    print("WEATHER ENCODING")
    print("="*60)

    # Create weather encoding BEFORE outlier detection
    weather_map = {'Sunny': 1, 'Cloudy': 2, 'Rainy': 3, 'Snowy': 4}
    df['weather_encoded'] = df['weather'].map(weather_map)

    # Fill any missing weather values with mode
    if df['weather_encoded'].isna().sum() > 0:
        df['weather_encoded'].fillna(df['weather_encoded'].mode()[0], inplace=True)

    print(f"Weather values mapping:")
    print(f"  Sunny: 1, Cloudy: 2, Rainy: 3, Snowy: 4")
    print(f"✓ Weather encoded successfully")

    # Drop original weather column (keep only encoded)
    df = df.drop('weather', axis=1)

    # ============================================================
    # IMPROVEMENT 2: DETECT AND HANDLE OUTLIERS
    # ============================================================
    print("\n" + "="*60)
    print("OUTLIER DETECTION")
    print("="*60)

    # Calculate IQR (Interquartile Range) for each series
    def detect_outliers_iqr(group, column='y', multiplier=1.5):
        Q1 = group[column].quantile(0.25)
        Q3 = group[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        return (group[column] < lower_bound) | (group[column] > upper_bound)

    # Identify outliers
    df['is_outlier'] = df.groupby('unique_id', group_keys=False).apply(
        lambda x: detect_outliers_iqr(x)
    )

    outliers_count = df['is_outlier'].sum()
    print(f"Total outliers detected: {outliers_count} / {len(df)} ({outliers_count/len(df)*100:.2f}%)")

    # Replace outliers with median of the series
    print("Replacing outliers with series median...")
    for uid in df['unique_id'].unique():
        mask = (df['unique_id'] == uid) & (df['is_outlier'])
        if mask.sum() > 0:
            median_val = df[df['unique_id'] == uid]['y'].median()
            df.loc[mask, 'y'] = median_val

    print("✓ Outliers replaced successfully")

    # ============================================================
    # IMPROVEMENT 3: ADD FEATURE ENGINEERING (SEASONALITY, CYCLICAL FEATURES)
    # ============================================================
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)

    # Extract date features
    df['day_of_week'] = df['ds'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_month'] = df['ds'].dt.day
    df['week_of_year'] = df['ds'].dt.isocalendar().week
    df['month'] = df['ds'].dt.month
    df['day_of_year'] = df['ds'].dt.dayofyear

    # Create cyclical encoding for day_of_week (sin/cos transformation)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Create cyclical encoding for month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Create weekend flag
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    print("✓ Features added:")
    print("  - Cyclical: day_of_week_sin, day_of_week_cos")
    print("  - Cyclical: month_sin, month_cos")
    print("  - Binary: is_weekend, holiday")
    print("  - Encoded: weather_encoded (1=Sunny, 2=Cloudy, 3=Rainy, 4=Snowy)")

    # Drop unnecessary columns
    df = df.drop(['day_of_week', 'day_of_month', 'week_of_year', 'month', 'day_of_year', 'is_outlier'], axis=1)

    print(f"\nFinal dataframe columns: {df.columns.tolist()}")

    # ============================================================
    # 2. TRAIN-TEST SPLIT FOR DECEMBER FORECAST
    # ============================================================
    # Train: Aug-Nov (approximately 120+ days)
    # Test: December (31 days)
    cutoff_date = pd.Timestamp('2023-11-30')

    Y_train_df = df[df['ds'] <= cutoff_date].copy()
    Y_test_df = df[df['ds'] > cutoff_date].copy()

    # Print detailed train and test information
    print("\n" + "="*60)
    print("TRAIN-TEST SPLIT INFO")
    print("="*60)
    print(f"\nTRAIN SET (August - November 2023):")
    print(f"  - Shape: {Y_train_df.shape}")
    print(f"  - Date range: {Y_train_df['ds'].min().strftime('%Y-%m-%d')} to {Y_train_df['ds'].max().strftime('%Y-%m-%d')}")
    print(f"  - Number of days: {(Y_train_df['ds'].max() - Y_train_df['ds'].min()).days + 1}")
    print(f"  - Total records: {len(Y_train_df)}")

    print(f"\nTEST SET (December 2023):")
    print(f"  - Shape: {Y_test_df.shape}")
    print(f"  - Date range: {Y_test_df['ds'].min().strftime('%Y-%m-%d')} to {Y_test_df['ds'].max().strftime('%Y-%m-%d')}")
    print(f"  - Number of days: {(Y_test_df['ds'].max() - Y_test_df['ds'].min()).days + 1}")
    print(f"  - Total records: {len(Y_test_df)}")