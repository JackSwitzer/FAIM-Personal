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

    # Add the following lines
    # Distributed training settings
    GPUS_PER_NODE = torch.cuda.device_count() if torch.cuda.is_available() else 0
    WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 1
    USE_DISTRIBUTED = WORLD_SIZE > 1

    # Data processing settings
    TARGET_VARIABLE = 'stock_exret'
    STANDARDIZE = True
    MIN_SEQUENCE_LENGTH = 5

    # Training settings
    NUM_EPOCHS = 100  # Increase epochs for full training
    BATCH_SIZE = 1024  # Increase batch size to utilize GPU memory
    LEARNING_RATE = 0.001
    ACCUMULATION_STEPS = 1 # Up 
    CLIP_GRAD_NORM = 1.0
    USE_ALL_DATA = True
    NUM_WORKERS = 4  # Adjust based on your CPU cores
    CHECKPOINT_INTERVAL = 100

    # LSTM hyperparameters 
    LSTM_PARAMS = {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout_rate': 0.2,
        'bidirectional': False,
        'use_batch_norm': True,
        'activation_function': 'LeakyReLU',
        'fc1_size': 64,
        'fc2_size': 32,
        'optimizer_name': 'Adam',
        'learning_rate': 0.001,
        'weight_decay': 1e-6,
        'seq_length': 10,
        'batch_size': BATCH_SIZE,
        'use_scheduler': True,
        'scheduler_factor': 0.5,
        'scheduler_patience': 5
    }

    # Hyperparameter optimization settings
    N_TRIALS = 25
    HYPEROPT_EPOCHS = 10  # Number of epochs during hyperparameter tuning

    # Logging settings
    LOG_INTERVAL = 5  # Log every 5 epochs

    # New option to use permco instead of permno
    USE_PERMCO = False  # Set this to True if you want to use permco instead of permno

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