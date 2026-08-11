 # src/data_ingestion.py
"""
Data ingestion pipeline for loading and validating data.
Supports multiple data sources: file, API, database.
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import settings
from src.logger_config import get_logger
from src.exceptions import (
    DataIngestionError,
    DataValidationError,
    FeatureNotFoundError
)

logger = get_logger()

class DataValidator:
    """
    Validates data quality and structure.
    """
    
    @staticmethod
    def validate_required_columns(
        data: pd.DataFrame,
        required_columns: List[str]
    ) -> None:
        """
        Validate that all required columns exist.
        
        Args:
            data: DataFrame to validate
            required_columns: List of required column names
        
        Raises:
            FeatureNotFoundError: If required columns are missing
        """
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            raise FeatureNotFoundError(
                f"Missing required columns: {missing_cols}"
            )
    
    @staticmethod
    def validate_nulls(
        data: pd.DataFrame,
        max_null_percentage: float = 5.0
    ) -> Dict[str, float]:
        """
        Validate null values in the data.
        
        Args:
            data: DataFrame to validate
            max_null_percentage: Maximum allowed null percentage
        
        Returns:
            Dictionary with null percentages per column
        
        Raises:
            DataValidationError: If null percentage exceeds threshold
        """
        null_percentages = (data.isnull().sum() / len(data)) * 100
        columns_with_nulls = null_percentages[null_percentages > 0]
        
        if not columns_with_nulls.empty:
            logger.warning(
                "Null values detected",
                extra={"null_percentages": columns_with_nulls.to_dict()}
            )
        
        # Check if any column exceeds threshold
        bad_columns = null_percentages[null_percentages > max_null_percentage]
        if not bad_columns.empty:
            raise DataValidationError(
                f"Columns with excessive null values: {bad_columns.to_dict()}"
            )
        
        return columns_with_nulls.to_dict()
    
    @staticmethod
    def validate_data_types(
        data: pd.DataFrame,
        numerical_features: List[str],
        categorical_features: List[str]
    ) -> Dict[str, str]:
        """
        Validate data types for features.
        
        Args:
            data: DataFrame to validate
            numerical_features: List of numerical features
            categorical_features: List of categorical features
        
        Returns:
            Dictionary of data type issues
        
        Raises:
            DataValidationError: If data type validation fails
        """
        issues = {}
        
        for feature in numerical_features:
            if feature in data.columns:
                if not pd.api.types.is_numeric_dtype(data[feature]):
                    issues[feature] = f"Expected numeric, got {data[feature].dtype}"
        
        for feature in categorical_features:
            if feature in data.columns:
                if pd.api.types.is_numeric_dtype(data[feature]):
                    issues[feature] = f"Expected categorical, got {data[feature].dtype}"
        
        if issues:
            logger.warning("Data type issues found", extra={"issues": issues})
        
        return issues
    
    @staticmethod
    def validate_value_ranges(
        data: pd.DataFrame,
        ranges: Dict[str, Dict[str, Union[int, float]]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate value ranges for numerical features.
        
        Args:
            data: DataFrame to validate
            ranges: Dictionary of min/max ranges per feature
        
        Returns:
            Dictionary of out-of-range values
        """
        violations = {}
        
        for feature, range_dict in ranges.items():
            if feature in data.columns:
                min_val = range_dict.get("min")
                max_val = range_dict.get("max")
                
                if min_val is not None:
                    below_min = data[data[feature] < min_val]
                    if not below_min.empty:
                        violations[feature] = {
                            "below_min": len(below_min),
                            "values": below_min[feature].head(5).tolist()
                        }
                
                if max_val is not None:
                    above_max = data[data[feature] > max_val]
                    if not above_max.empty:
                        if feature not in violations:
                            violations[feature] = {}
                        violations[feature]["above_max"] = len(above_max)
                        violations[feature]["values"] = above_max[feature].head(5).tolist()
        
        return violations

class DataIngestion:
    """
    Handles loading and validation of data from various sources.
    """
    
    def __init__(self):
        self.reference_data: Optional[pd.DataFrame] = None
        self.incoming_data: Optional[pd.DataFrame] = None
        self.data_metadata: Dict[str, Any] = {}
        self.validator = DataValidator()
        
        # Set up HTTP session for API calls
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """
        Create HTTP session with retry logic.
        
        Returns:
            Configured requests session
        """
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def load_reference_data(self) -> pd.DataFrame:
        """
        Load reference training data.
        
        Returns:
            Reference data as DataFrame
        
        Raises:
            DataIngestionError: If loading fails
        """
        try:
            logger.info(
                "Loading reference data",
                extra={"path": settings.REFERENCE_DATA_PATH}
            )
            
            if not os.path.exists(settings.REFERENCE_DATA_PATH):
                raise DataIngestionError(
                    f"Reference data not found: {settings.REFERENCE_DATA_PATH}"
                )
            
            self.reference_data = pd.read_csv(settings.REFERENCE_DATA_PATH)
            
            # Validate reference data
            self._validate_reference_data()
            
            # Store metadata
            self._store_metadata("reference")
            
            logger.info(
                "Reference data loaded successfully",
                extra={
                    "rows": len(self.reference_data),
                    "columns": len(self.reference_data.columns)
                }
            )
            
            return self.reference_data
            
        except Exception as e:
            logger.error(f"Failed to load reference data: {str(e)}")
            raise DataIngestionError(f"Failed to load reference data: {str(e)}")
    
    def _validate_reference_data(self) -> None:
        """Validate reference data structure and quality."""
        if self.reference_data is None:
            raise DataValidationError("Reference data not loaded")
        
        # Check required columns
        all_features = (
            settings.NUMERICAL_FEATURES +
            settings.CATEGORICAL_FEATURES +
            [settings.TARGET_COLUMN]
        )
        self.validator.validate_required_columns(
            self.reference_data,
            all_features
        )
        
        # Check for null values
        self.validator.validate_nulls(self.reference_data)
        
        # Check data types
        self.validator.validate_data_types(
            self.reference_data,
            settings.NUMERICAL_FEATURES,
            settings.CATEGORICAL_FEATURES
        )
    
    def load_incoming_data(
        self,
        source: str = "file",
        **kwargs
    ) -> pd.DataFrame:
        """
        Load incoming/production data from various sources.
        
        Args:
            source: Data source type ('file', 'api', 'database')
            **kwargs: Source-specific arguments
        
        Returns:
            Incoming data as DataFrame
        
        Raises:
            DataIngestionError: If loading fails
        """
        try:
            logger.info(
                "Loading incoming data",
                extra={"source": source}
            )
            
            if source == "file":
                self.incoming_data = self._load_from_file(**kwargs)
            elif source == "api":
                self.incoming_data = self._load_from_api(**kwargs)
            elif source == "database":
                self.incoming_data = self._load_from_database(**kwargs)
            else:
                raise ValueError(f"Unsupported data source: {source}")
            
            # Validate incoming data
            if self.incoming_data is not None:
                self._validate_incoming_data()
                
                # Store metadata
                self._store_metadata("incoming", source=source)
                
                logger.info(
                    "Incoming data loaded successfully",
                    extra={
                        "rows": len(self.incoming_data),
                        "columns": len(self.incoming_data.columns),
                        "source": source
                    }
                )
            
            return self.incoming_data
            
        except Exception as e:
            logger.error(f"Failed to load incoming data: {str(e)}")
            raise DataIngestionError(f"Failed to load incoming data: {str(e)}")
    
    def _load_from_file(
        self,
        file_path: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from a file.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional pandas read_csv arguments
        
        Returns:
            DataFrame from file
        """
        if file_path is None:
            # Find latest CSV file in incoming directory
            incoming_dir = Path(settings.INCOMING_DATA_PATH)
            csv_files = list(incoming_dir.glob("*.csv"))
            
            if not csv_files:
                raise FileNotFoundError(
                    f"No CSV files found in {settings.INCOMING_DATA_PATH}"
                )
            
            file_path = str(max(csv_files, key=lambda f: f.stat().st_ctime))
            logger.info(f"Using latest file: {file_path}")
        
        return pd.read_csv(file_path, **kwargs)
    
    def _load_from_api(
        self,
        api_url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from a REST API.
        
        Args:
            api_url: API endpoint URL
            headers: HTTP headers
            params: Query parameters
        
        Returns:
            DataFrame from API response
        
        Raises:
            DataIngestionError: If API request fails
        """
        try:
            headers = headers or {}
            params = params or {}
            
            response = self.session.get(
                api_url,
                headers=headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Handle different response formats
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Common API response formats
                for key in ["data", "results", "items", "records"]:
                    if key in data:
                        return pd.DataFrame(data[key])
                
                # Try to find array in response
                for value in data.values():
                    if isinstance(value, list) and len(value) > 0:
                        return pd.DataFrame(value)
                
                return pd.DataFrame([data])
            else:
                raise DataIngestionError(
                    f"Unexpected API response format: {type(data)}"
                )
                
        except requests.exceptions.RequestException as e:
            raise DataIngestionError(f"API request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise DataIngestionError(f"Invalid JSON response: {str(e)}")
    
    def _load_from_database(self, **kwargs) -> pd.DataFrame:
        """
        Load data from a database.
        
        This is a placeholder implementation.
        Override with actual database connection logic.
        
        Args:
            **kwargs: Database connection parameters
        
        Returns:
            DataFrame from database
        
        Raises:
            NotImplementedError: Not implemented in base class
        """
        raise NotImplementedError(
            "Database loading not implemented. "
            "Override this method with database-specific logic."
        )
    
    def _validate_incoming_data(self) -> None:
        """Validate incoming data structure and quality."""
        if self.incoming_data is None:
            raise DataValidationError("Incoming data not loaded")
        
        # Check for empty DataFrame
        if len(self.incoming_data) == 0:
            raise DataValidationError("Incoming data is empty")
        
        # Check required columns
        all_features = (
            settings.NUMERICAL_FEATURES +
            settings.CATEGORICAL_FEATURES +
            [settings.TARGET_COLUMN]
        )
        
        try:
            self.validator.validate_required_columns(
                self.incoming_data,
                all_features
            )
        except FeatureNotFoundError:
            # Try with just the feature columns (no target)
            feature_columns = (
                settings.NUMERICAL_FEATURES +
                settings.CATEGORICAL_FEATURES
            )
            self.validator.validate_required_columns(
                self.incoming_data,
                feature_columns
            )
            logger.warning("Target column missing in incoming data")
        
        # Check for null values
        self.validator.validate_nulls(self.incoming_data)
        
        # Check data types
        self.validator.validate_data_types(
            self.incoming_data,
            settings.NUMERICAL_FEATURES,
            settings.CATEGORICAL_FEATURES
        )
    
    def _store_metadata(
        self,
        data_type: str,
        **kwargs
    ) -> None:
        """
        Store metadata about loaded data.
        
        Args:
            data_type: Type of data ('reference' or 'incoming')
            **kwargs: Additional metadata
        """
        data = getattr(self, f"{data_type}_data")
        if data is None:
            return
        
        self.data_metadata[data_type] = {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": data.dtypes.astype(str).to_dict(),
            "rows": len(data),
            "load_time": datetime.now().isoformat(),
            **kwargs
        }
    
    def save_incoming_data(
        self,
        data: Optional[pd.DataFrame] = None,
        filename: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Save incoming data for historical tracking.
        
        Args:
            data: DataFrame to save
            filename: Custom filename (auto-generated if not provided)
            **kwargs: Additional pandas to_csv arguments
        
        Returns:
            Path to saved file
        """
        if data is None:
            data = self.incoming_data
        
        if data is None:
            raise ValueError("No data to save")
        
        # Generate filename with timestamp
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"incoming_data_{timestamp}.csv"
        
        filepath = Path(settings.INCOMING_DATA_PATH) / filename
        data.to_csv(filepath, index=False, **kwargs)
        
        logger.info(f"Saved incoming data to {filepath}")
        return str(filepath)
    
    def get_data_statistics(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the data.
        
        Args:
            data: DataFrame to analyze
        
        Returns:
            Dictionary of statistics
        """
        if data is None:
            data = self.incoming_data
        
        if data is None:
            return {}
        
        stats = {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "null_count": data.isnull().sum().sum(),
            "null_percentage": (data.isnull().sum().sum() / data.size) * 100,
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024**2,
            "duplicate_rows": data.duplicated().sum()
        }
        
        # Numerical features statistics
        numerical_stats = {}
        for feature in settings.NUMERICAL_FEATURES:
            if feature in data.columns:
                col_data = data[feature].dropna()
                if not col_data.empty:
                    numerical_stats[feature] = {
                        "mean": float(col_data.mean()),
                        "std": float(col_data.std()),
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "q25": float(col_data.quantile(0.25)),
                        "q50": float(col_data.quantile(0.50)),
                        "q75": float(col_data.quantile(0.75))
                    }
        
        stats["numerical_stats"] = numerical_stats
        
        # Categorical features statistics
        categorical_stats = {}
        for feature in settings.CATEGORICAL_FEATURES:
            if feature in data.columns:
                col_data = data[feature].dropna()
                if not col_data.empty:
                    value_counts = col_data.value_counts()
                    categorical_stats[feature] = {
                        "unique_values": len(value_counts),
                        "most_common": value_counts.head(5).to_dict(),
                        "missing_count": data[feature].isnull().sum()
                    }
        
        stats["categorical_stats"] = categorical_stats
        
        return stats
    
    def calculate_data_hash(self, data: Optional[pd.DataFrame] = None) -> str:
        """
        Calculate a hash of the data for version tracking.
        
        Args:
            data: DataFrame to hash
        
        Returns:
            MD5 hash string
        """
        if data is None:
            data = self.incoming_data
        
        if data is None:
            return ""
        
        # Create a string representation of the data
        data_string = data.to_csv(index=False).encode()
        return hashlib.md5(data_string).hexdigest()
