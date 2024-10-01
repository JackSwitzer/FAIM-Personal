import os
import random
import numpy as np
import torch
import logging
import gc
import sys
import importlib
import pynvml
from torch.distributed import ReduceOp
from packaging import version
from datetime import datetime
from config import Config
from logging.handlers import RotatingFileHandler

# Create a global logger
logger = logging.getLogger('stock_predictor')
logger.setLevel(logging.INFO)

def setup_logging(log_dir, log_filename=None):
    """Set up logging configuration with a unique log file name."""
    if log_filename is None:
        log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, 'logs')
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, log_filename)
    
    # Get the root logger
    logger = logging.getLogger()
    
    # Check if handlers are already set up to avoid duplicate logs
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler with rotation
        fh = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    logger.info(f"Logging initialized. Log file: {log_file}")

def get_logger(name=None):
    """Retrieve the global logger instance."""
    return logging.getLogger(name)

def save_csv(df, output_dir, filename):
    """Save the DataFrame to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    logger.info(f"Data saved to: {output_path}")

def calculate_oos_r2(y_true, y_pred):
    """Calculate Out-of-Sample R-squared."""
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum(y_true ** 2)
    if denominator == 0:
        logger.warning("Denominator in OOS R^2 calculation is zero. Returning NaN.")
        return np.nan
    r2 = 1 - numerator / denominator
    return r2

def clear_gpu_memory():
    """Clear GPU memory by deleting unnecessary variables and emptying the cache."""
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) or (hasattr(obj, 'data') and isinstance(obj.data, torch.Tensor)):
                del obj
        except:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def clear_import_cache():
    """Clear the Python import cache by reloading all modules."""
    for module in list(sys.modules.keys()):
        if module not in sys.builtin_module_names:
            importlib.reload(sys.modules[module])
    logger.info("Python import cache cleared.")

def check_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA. GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Metal Performance Shaders) on macOS.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU.")
    return device

def log_gpu_memory_usage():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logger.info(f"GPU memory: used={info.used/1024**2:.1f}MB, free={info.free/1024**2:.1f}MB, total={info.total/1024**2:.1f}MB")
    except pynvml.NVMLError as e:
        logger.warning(f"Unable to log GPU memory usage: {e}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as e:
            logger.warning(f"Error shutting down NVML: {e}")

def check_torch_version():
    required_version = "1.7.0"  # Adjust this to the minimum required version
    current_version = torch.__version__
    logger.info(f"Using PyTorch version: {current_version}")
    if version.parse(current_version) < version.parse(required_version):
        logger.warning(f"PyTorch version {current_version} is older than the recommended version {required_version}. Some features may not work as expected.")

def log_memory_usage():
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")

def log_gpu_memory():
    if torch.cuda.is_available():
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def set_seed(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)