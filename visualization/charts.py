import matplotlib.pyplot as plt


def plot_forecast_comparison(df_train, df_test, forecasts, output_path):
    forecasts_reset = forecasts.reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_train['ds'], df_train['y'], label='Actual 2023 (Train)', color='blue')
    plt.plot(df_test['ds'], df_test['y'], label='Actual Jan 2024 (Test)', color='green')
    plt.plot(forecasts_reset['ds'], forecasts_reset['NHITS'], 
             label='NHITS Forecast', color='red', linestyle='--')
    
    plt.fill_between(forecasts_reset['ds'], 
                     forecasts_reset['NHITS-lo-90'], 
                     forecasts_reset['NHITS-hi-90'], 
                     color='red', alpha=0.2, label='90% Confidence Interval')
    
    plt.title('Fashion Sales Forecast: Actual vs Predicted (NHITS)')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"Chart saved at {output_path}")


def display_comparison_results(comparison, n=5):
    print(f"\nComparison of first {n} days of January 2024:")
    print(comparison.head(n))

