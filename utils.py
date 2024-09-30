import os
import random
import numpy as np
import torch
import logging
import gc
import sys
import importlib
import pynvml
from packaging import version
from datetime import datetime

def setup_logging(log_dir, level=logging.INFO):
    """Set up logging configuration with a unique log file name."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")

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

def check_cuda():
    """Check if CUDA is available and return the appropriate device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logging.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("CUDA is not available. Using CPU.")
    logging.info(f"PyTorch version: {torch.__version__}")
    return device

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

def log_memory_usage():
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    logging.info(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")

def log_gpu_memory():
    if torch.cuda.is_available():
        logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logging.info(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)