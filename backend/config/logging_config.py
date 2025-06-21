"""
Logging configuration for the 2K Flash application.
"""

import logging
import logging.handlers
import os
from pathlib import Path

# Base log directory
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Log file paths
API_LOG = LOG_DIR / "api.log"
DASHBOARD_LOG = LOG_DIR / "dashboard.log"
DATA_FETCHER_LOG = LOG_DIR / "data_fetcher.log"

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Maximum log file size (10 MB)
MAX_LOG_SIZE = 10 * 1024 * 1024

# Number of backup log files
BACKUP_COUNT = 5


def configure_logger(name, log_file, level=logging.INFO):
    """
    Configure a logger with file and console handlers.
    
    Args:
        name (str): Logger name
        log_file (Path): Log file path
        level (int): Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    
    # Create file handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Configure loggers for different components
def get_api_logger():
    return configure_logger("api", API_LOG)


def get_dashboard_logger():
    return configure_logger("dashboard", DASHBOARD_LOG)


def get_data_fetcher_logger():
    return configure_logger("data_fetcher", DATA_FETCHER_LOG)
