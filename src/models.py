from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import DistributionLoss


class ModelConfig:
    HORIZON = 31
    INPUT_SIZE = 90
    CONFIDENCE_LEVELS = [80, 90]
    FREQ_DOWNSAMPLE = [4, 2, 1]
    SCALER_TYPE = 'robust'
    MAX_STEPS = 1000
    LEARNING_RATE = 5e-4
    VAL_CHECK_STEPS = 50
    EARLY_STOP_PATIENCE_STEPS = 20
    BATCH_SIZE = 64


def create_nhits_model(
    horizon=ModelConfig.HORIZON,
    input_size=ModelConfig.INPUT_SIZE,
    levels=ModelConfig.CONFIDENCE_LEVELS,
    max_steps=ModelConfig.MAX_STEPS,
    learning_rate=ModelConfig.LEARNING_RATE,
    val_check_steps=ModelConfig.VAL_CHECK_STEPS,
    early_stop_patience_steps=ModelConfig.EARLY_STOP_PATIENCE_STEPS,
    batch_size=ModelConfig.BATCH_SIZE,
):
    model = NHITS(
        h=horizon,
        input_size=input_size,
        loss=DistributionLoss(distribution='Poisson', level=levels),
        n_freq_downsample=ModelConfig.FREQ_DOWNSAMPLE,
        scaler_type=ModelConfig.SCALER_TYPE,
        max_steps=max_steps,
        learning_rate=learning_rate,
        val_check_steps=val_check_steps,
        early_stop_patience_steps=early_stop_patience_steps,
        batch_size=batch_size,
    )
    return model

