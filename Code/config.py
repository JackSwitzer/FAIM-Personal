import os
import torch

class Config:
    # General settings
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUT_DIR = r"C:\Users\jacks\Documents\Code\McGill FAIM\Data Output"
    MODEL_WEIGHTS_DIR = os.path.join(OUT_DIR, "model_weights")
    DATA_INPUT_DIR = r"C:\Users\jacks\Documents\Code\McGill FAIM\Data Input"
    FULL_DATA_PATH = os.path.join(DATA_INPUT_DIR, "hackathon_sample_v2.csv")

    # Data processing settings
    TARGET_VARIABLE = 'stock_exret'
    STANDARDIZE = True

    # Training settings
    NUM_EPOCHS = 1000
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    ACCUMULATION_STEPS = 1
    CLIP_GRAD_NORM = 1.0
    USE_ALL_DATA = True
    NUM_WORKERS = 2

    # LSTM model settings
    LSTM_PARAMS = {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout_rate': 0.2,
        'bidirectional': False,
        'use_batch_norm': True,
        'activation_function': 'LeakyReLU',
        'fc1_size': 64,
        'fc2_size': 32
    }

    # Hyperparameter optimization settings
    N_TRIALS = 25

    # Logging settings
    LOG_INTERVAL = 5  # Log every 5 epochs

    @classmethod
    def update(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise AttributeError(f"Config has no attribute '{key}'")

    @classmethod
    def get_lstm_params(cls):
        return cls.LSTM_PARAMS.copy()