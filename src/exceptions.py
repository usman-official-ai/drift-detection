 # src/exceptions.py
"""
Custom exceptions for the drift detection system.
"""

class DriftDetectionError(Exception):
    """Base exception for drift detection system."""
    pass

class DataIngestionError(DriftDetectionError):
    """Raised when data ingestion fails."""
    pass

class DataValidationError(DriftDetectionError):
    """Raised when data validation fails."""
    pass

class ConfigurationError(DriftDetectionError):
    """Raised when configuration is invalid."""
    pass

class DriftDetectionError(DriftDetectionError):
    """Raised when drift detection fails."""
    pass

class AlertDeliveryError(DriftDetectionError):
    """Raised when alert delivery fails."""
    pass

class FeatureNotFoundError(DriftDetectionError):
    """Raised when required features are missing."""
    pass
