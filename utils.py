import os
import numpy as np
import torch
import logging
import gc
import sys
import importlib
import argparse
import pynvml
from packaging import version

def save_csv(df, output_dir, filename):
    """Save the DataFrame to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    logging.info(f"Data saved to: {output_path}")

def calculate_oos_r2(y_true, y_pred):
    """Calculate Out-of-Sample R-squared."""
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum(y_true ** 2)
    if denominator == 0:
        logging.warning("Denominator in OOS R^2 calculation is zero. Returning NaN.")
        return np.nan
    r2 = 1 - numerator / denominator
    return r2

def clear_gpu_memory():
    """Clear GPU memory by deleting unnecessary variables and emptying the cache."""
    torch.cuda.empty_cache()
    gc.collect()

def clear_import_cache():
    """Clear the Python import cache by reloading all modules."""
    for module in list(sys.modules.keys()):
        if module not in sys.builtin_module_names:
            importlib.reload(sys.modules[module])
    logging.info("Python import cache cleared.")

def setup_logging(log_file, level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def check_cuda():
    """Check for CUDA availability and return the appropriate device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logging.info("CUDA is not available. Using CPU.")
    logging.info(f"PyTorch version: {torch.__version__}")
    return device

def parse_args():
    parser = argparse.ArgumentParser(description='Stock Prediction using LSTM')
    parser.add_argument('--data_input_dir', type=str, default=r"C:\Users\jacks\Documents\Code\McGill FAIM\Data Input", help='Input data directory')
    parser.add_argument('--out_dir', type=str, default=r"C:\Users\jacks\Documents\Code\McGill FAIM\Data Output", help='Output directory')
    parser.add_argument('--data_file', type=str, default="hackathon_sample_v2.csv", help='Data file name')
    parser.add_argument('--optimize', action='store_true', help='Whether to perform hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of trials for hyperparameter optimization')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq_length', type=int, default=10, help='Sequence length for LSTM')
    return parser.parse_args()

def log_gpu_memory_usage():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logging.info(f"GPU memory: used={info.used/1024**2:.1f}MB, free={info.free/1024**2:.1f}MB, total={info.total/1024**2:.1f}MB")
    except:
        logging.warning("Unable to log GPU memory usage")
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

def check_torch_version():
    required_version = "1.7.0"  # Adjust this to the minimum required version
    current_version = torch.__version__
    if version.parse(current_version) < version.parse(required_version):
        logging.warning(f"PyTorch version {current_version} is older than the recommended version {required_version}. Some features may not work as expected.")

# Call this function periodically in your training loop
