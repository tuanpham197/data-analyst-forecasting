from neuralforecast import NeuralForecast
from data.generate_data import generate_fashion_data
from src.data_processing import split_train_test, merge_forecast_with_actual, smooth_data
from src.models import create_nhits_model
from visualization.charts import plot_forecast_comparison, display_comparison_results
from evaluation.metrics import evaluate_forecast


class ForecastPipeline:
    def __init__(self, freq='D', test_start_date='2024-01-01', output_path='reports/forecast_comparison.png', 
                 apply_smoothing=False, smoothing_method='exponential', smoothing_alpha=0.3):
        self.freq = freq
        self.test_start_date = test_start_date
        self.output_path = output_path
        self.apply_smoothing = apply_smoothing
        self.smoothing_method = smoothing_method
        self.smoothing_alpha = smoothing_alpha
        self.df = None
        self.df_train = None
        self.df_test = None
        self.forecasts = None
        self.nf = None
    
    def load_data(self):
        print("Loading data...")
        self.df = generate_fashion_data()
        return self
    
    def prepare_data(self):
        print("\nPreparing data...")
        self.df_train, self.df_test = split_train_test(self.df, self.test_start_date)
        
        if self.apply_smoothing:
            print(f"Applying {self.smoothing_method} smoothing to training data...")
            self.df_train = smooth_data(self.df_train, method=self.smoothing_method, alpha=self.smoothing_alpha)
            print("Smoothing applied successfully!")
        
        return self
    
    def train_model(self, **model_kwargs):
        print("\nTraining model...")
        print(f"Configuration: max_steps={model_kwargs.get('max_steps', 'default')}, "
              f"learning_rate={model_kwargs.get('learning_rate', 'default')}, "
              f"input_size={model_kwargs.get('input_size', 'default')}")
        model = create_nhits_model(**model_kwargs)
        self.nf = NeuralForecast(models=[model], freq=self.freq)
        self.nf.fit(df=self.df_train, val_size=31)
        print("Training completed!")
        return self
    
    def predict(self):
        print("\nGenerating forecasts...")
        self.forecasts = self.nf.predict()
        return self
    
    def visualize(self):
        print("\nCreating visualization...")
        plot_forecast_comparison(self.df_train, self.df_test, self.forecasts, self.output_path)
        return self
    
    def evaluate(self):
        print("\nEvaluating results...")
        metrics, comparison = evaluate_forecast(self.df_test, self.forecasts)
        
        print("\n" + "="*50)
        print("FORECAST EVALUATION METRICS")
        print("="*50)
        print(f"MAE  (Mean Absolute Error):      {metrics['MAE']:.2f}")
        print(f"RMSE (Root Mean Squared Error):  {metrics['RMSE']:.2f}")
        print(f"MAPE (Mean Absolute % Error):    {metrics['MAPE']:.2f}%")
        print("="*50)
        
        if metrics['MAPE'] < 10.0:
            print("✅ Target achieved! MAPE < 10%")
        else:
            print(f"⚠️  Target not met. Need MAPE < 10% (current: {metrics['MAPE']:.2f}%)")
        
        display_comparison_results(comparison)
        return comparison, metrics
    
    def run(self, **model_kwargs):
        self.load_data()
        self.prepare_data()
        self.train_model(**model_kwargs)
        self.predict()
        self.visualize()
        comparison, metrics = self.evaluate()
        return comparison, metrics

