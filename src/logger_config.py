 # src/logger_config.py
"""
Logging configuration with JSON formatting.
Supports both console and file logging with different formats.
"""

import logging
import sys
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Dict, Any
from pythonjsonlogger import jsonlogger

from src.config import settings
from src.exceptions import ConfigurationError

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter for structured logging.
    Adds additional fields to log records.
    """
    
    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any]
    ) -> None:
        """
        Add custom fields to log record.
        
        Args:
            log_record: Dictionary to add fields to
            record: Original log record
            message_dict: Additional message dictionary
        """
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record["timestamp"] = datetime.utcnow().isoformat()
        
        # Add log level
        log_record["level"] = record.levelname
        
        # Add module and function information
        log_record["module"] = record.module
        log_record["function"] = record.funcName
        log_record["line"] = record.lineno
        
        # Add process and thread information
        log_record["process_id"] = record.process
        log_record["thread_id"] = record.thread
        
        # Remove default fields
        log_record.pop("asctime", None)
        log_record.pop("exc_info", None)
        log_record.pop("exc_text", None)

class LoggerFactory:
    """
    Factory class for creating and configuring loggers.
    """
    
    @staticmethod
    def create_logger(
        name: str = "drift_detector",
        log_file: Optional[str] = None,
        log_level: str = "INFO",
        json_format: bool = False,
        max_bytes: int = 10485760,  # 10MB
        backup_count: int = 5
    ) -> logging.Logger:
        """
        Create and configure a logger.
        
        Args:
            name: Logger name
            log_file: Path to log file
            log_level: Logging level
            json_format: Use JSON format
            max_bytes: Maximum size of log file before rotation
            backup_count: Number of backup files to keep
        
        Returns:
            Configured logger instance
        """
        # Get logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatters
        if json_format:
            file_formatter = CustomJsonFormatter()
            console_formatter = CustomJsonFormatter()
        else:
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        if log_file:
            try:
                # Ensure log directory exists
                log_dir = os.path.dirname(log_file)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                
                # Use TimedRotatingFileHandler for daily rotation
                file_handler = TimedRotatingFileHandler(
                    filename=log_file,
                    when="midnight",
                    interval=1,
                    backupCount=backup_count
                )
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
                
                # Error file handler
                error_log_file = log_file.replace(".log", "_errors.log")
                error_handler = TimedRotatingFileHandler(
                    filename=error_log_file,
                    when="midnight",
                    interval=1,
                    backupCount=backup_count
                )
                error_handler.setLevel(logging.ERROR)
                error_handler.setFormatter(file_formatter)
                logger.addHandler(error_handler)
                
            except Exception as e:
                console_handler.setLevel(logging.ERROR)
                console_handler.setFormatter(
                    logging.Formatter("%(asctime)s - ERROR - %(message)s")
                )
                logger.error(f"Failed to create file handlers: {str(e)}")
        
        return logger

# Create global logger
logger = LoggerFactory.create_logger(
    name="drift_detector",
    log_file=settings.LOGS_PATH,
    log_level=settings.LOG_LEVEL,
    json_format=False  # Set to True for production
)

def get_logger() -> logging.Logger:
    """
    Get the global logger instance.
    
    Returns:
        Logger instance
    """
    return logger
