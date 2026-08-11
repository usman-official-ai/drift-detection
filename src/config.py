 # src/config.py
"""
Configuration management using Pydantic settings.
Loads configuration from environment variables and .env file.
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    All settings can be overridden via environment variables.
    """
    
    # Data Paths
    REFERENCE_DATA_PATH: str = Field(
        default="data/reference/training_data.csv",
        description="Path to reference training data"
    )
    INCOMING_DATA_PATH: str = Field(
        default="data/incoming/",
        description="Directory for incoming data"
    )
    REPORTS_PATH: str = Field(
        default="reports/",
        description="Directory for drift reports"
    )
    LOGS_PATH: str = Field(
        default="logs/app.log",
        description="Path for application logs"
    )
    
    # Drift Thresholds
    PSI_THRESHOLD: float = Field(
        default=0.2,
        description="PSI threshold for drift detection",
        ge=0.0,
        le=1.0
    )
    KS_THRESHOLD: float = Field(
        default=0.05,
        description="KS test p-value threshold",
        ge=0.0,
        le=1.0
    )
    DRIFT_PERCENTAGE_THRESHOLD: float = Field(
        default=0.3,
        description="Percentage of drifting features to trigger alert",
        ge=0.0,
        le=1.0
    )
    
    # Feature Configuration
    NUMERICAL_FEATURES: List[str] = Field(
        default=["age", "income", "transaction_amount", "credit_score"],
        description="List of numerical features"
    )
    CATEGORICAL_FEATURES: List[str] = Field(
        default=["gender", "occupation", "location", "product_category"],
        description="List of categorical features"
    )
    TARGET_COLUMN: str = Field(
        default="target",
        description="Target column name"
    )
    IGNORE_COLUMNS: List[str] = Field(
        default=["id", "timestamp", "date"],
        description="Columns to ignore in drift detection"
    )
    
    # Alert Settings
    ENABLE_EMAIL_ALERTS: bool = Field(
        default=True,
        description="Enable email alerts"
    )
    ENABLE_SLACK_ALERTS: bool = Field(
        default=True,
        description="Enable Slack alerts"
    )
    
    # Email Configuration
    SMTP_SERVER: str = Field(
        default="smtp.gmail.com",
        description="SMTP server hostname"
    )
    SMTP_PORT: int = Field(
        default=587,
        description="SMTP server port"
    )
    EMAIL_SENDER: Optional[str] = Field(
        default=None,
        description="Sender email address"
    )
    EMAIL_PASSWORD: Optional[str] = Field(
        default=None,
        description="Email password or app-specific password"
    )
    EMAIL_RECIPIENTS: List[str] = Field(
        default=["admin@example.com"],
        description="List of email recipients"
    )
    EMAIL_CC: Optional[List[str]] = Field(
        default=None,
        description="CC email recipients"
    )
    EMAIL_BCC: Optional[List[str]] = Field(
        default=None,
        description="BCC email recipients"
    )
    
    # Slack Configuration
    SLACK_WEBHOOK_URL: Optional[str] = Field(
        default=None,
        description="Slack webhook URL"
    )
    SLACK_CHANNEL: Optional[str] = Field(
        default=None,
        description="Slack channel name"
    )
    SLACK_USERNAME: str = Field(
        default="Drift Detector",
        description="Slack bot username"
    )
    SLACK_ICON_EMOJI: str = Field(
        default=":warning:",
        description="Slack bot icon emoji"
    )
    
    # System Configuration
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    TIMEZONE: str = Field(
        default="UTC",
        description="System timezone"
    )
    SCHEDULE_INTERVAL_HOURS: int = Field(
        default=24,
        description="Schedule interval in hours"
    )
    SCHEDULE_START_TIME: str = Field(
        default="06:00",
        description="Schedule start time (HH:MM)"
    )
    
    # Performance Settings
    SAMPLE_SIZE: Optional[int] = Field(
        default=None,
        description="Sample size for drift detection (None for all data)"
    )
    BATCH_SIZE: int = Field(
        default=10000,
        description="Batch size for processing large datasets"
    )
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()
    
    @validator("EMAIL_RECIPIENTS", pre=True)
    def parse_email_recipients(cls, v):
        """Parse email recipients from string or list."""
        if isinstance(v, str):
            return [email.strip() for email in v.split(",") if email.strip()]
        return v
    
    @validator("NUMERICAL_FEATURES", "CATEGORICAL_FEATURES", pre=True)
    def parse_features(cls, v):
        """Parse features from string or list."""
        if isinstance(v, str):
            return [feature.strip() for feature in v.split(",") if feature.strip()]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"
    
    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        directories = [
            os.path.dirname(self.REPORTS_PATH),
            os.path.dirname(self.LOGS_PATH),
            self.INCOMING_DATA_PATH,
            os.path.dirname(self.REFERENCE_DATA_PATH)
        ]
        
        for directory in directories:
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
    
    def validate_configuration(self) -> List[str]:
        """
        Validate configuration and return list of errors.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check reference data exists
        if not os.path.exists(self.REFERENCE_DATA_PATH):
            errors.append(f"Reference data not found: {self.REFERENCE_DATA_PATH}")
        
        # Validate email configuration
        if self.ENABLE_EMAIL_ALERTS:
            if not self.EMAIL_SENDER:
                errors.append("EMAIL_SENDER is required when email alerts are enabled")
            if not self.EMAIL_PASSWORD:
                errors.append("EMAIL_PASSWORD is required when email alerts are enabled")
            if not self.EMAIL_RECIPIENTS:
                errors.append("EMAIL_RECIPIENTS is required when email alerts are enabled")
        
        # Validate Slack configuration
        if self.ENABLE_SLACK_ALERTS:
            if not self.SLACK_WEBHOOK_URL:
                errors.append("SLACK_WEBHOOK_URL is required when Slack alerts are enabled")
        
        return errors

# Create global settings instance
settings = Settings()

# Ensure directories exist
settings.ensure_directories()
