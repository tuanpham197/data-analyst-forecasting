import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from src.pipeline import ForecastPipeline


def main():
    pipeline = ForecastPipeline(
        freq='D',
        test_start_date='2024-01-01',
        output_path='reports/forecast_comparison.png',
        apply_smoothing=True,
        smoothing_method='exponential',
        smoothing_alpha=0.3
    )
    
    pipeline.run(
        horizon=31,
        input_size=90,
        max_steps=500,
        learning_rate=5e-4,
        val_check_steps=50
    )


if __name__ == '__main__':
    main()