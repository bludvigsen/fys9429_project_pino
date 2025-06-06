"""Logging configuration for the PINO surrogate model application."""
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(level=logging.INFO, log_to_file=True):
    """Set up logging for the application.
    
    Args:
        level: Logging level
        log_to_file: Whether to log to a file
        
    Returns:
        None
    """
    # Create logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler if requested
    if log_to_file:
        # Create logs directory if it doesn't exist
        log_dir = Path(__file__).parent.parent.parent / 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'pino_surrogate_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logging.info(f"Logging initialized at level {level}")
    return root_logger 