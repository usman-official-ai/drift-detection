 # tests/test_drift_detector.py
"""
Unit tests for drift detection system.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.drift_detector import DriftDetector, StatisticalTests
from src.data_ingestion import DataIngestion, DataValidator
from src.alert_manager import AlertManager, EmailAlert, SlackAlert
from src.config import settings
from src.exceptions import DataIngestionError, DataValidationError

class TestStatisticalTests:
    """Test statistical test functions."""
    
    def test_ks_test(self):
        """Test Kolmogorov-Smirnov test."""
        # Similar distributions
        sample1 = np.random.normal(0, 1, 1000)
        sample2 = np.random.normal(0.1, 1, 1000)
        
        statistic, p_value = StatisticalTests.ks_test(
            pd.Series(sample1),
            pd.Series(sample2)
        )
        
        assert 0 <= statistic <= 1
        assert 0 <= p_value <= 1
    
    def test_psi(self):
        """Test PSI calculation."""
        # Same distribution
        expected = pd.Series(['A', 'B', 'C'] * 100)
        actual = pd.Series(['A', 'B', 'C'] * 100)
        
        psi = StatisticalTests.psi(expected, actual)
        assert psi == pytest.approx(0.0, abs=0.01)
        
        # Different distribution
        expected = pd.Series(['A', 'B', 'C'] * 100)
        actual = pd.Series(['A', 'B', 'C', 'D'] * 75)
        
        psi = StatisticalTests.psi(expected, actual)
        assert psi > 0.1

class TestDataValidator:
    """Test data validation functions."""
    
    def test_validate_required_columns(self):
        """Test required columns validation."""
        data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        required = ['col1', 'col2']
        
        # Should not raise
        DataValidator.validate_required_columns(data, required)
        
        # Should raise
        required = ['col1', 'col2', 'col3']
        with pytest.raises(Exception):
            DataValidator.validate_required_columns(data, required)
    
    def test_validate_nulls(self):
        """Test null value validation."""
        data = pd.DataFrame({
            'col1': [1, 2, None],
            'col2': [4, 5, 6]
        })
        
        result = DataValidator.validate_nulls(data, max_null_percentage=50)
        assert 'col1' in result

class TestDataIngestion:
    """Test data ingestion functionality."""
    
    @patch('src.data_ingestion.pd.read_csv')
    def test_load_reference_data(self, mock_read_csv):
        """Test loading reference data."""
        mock_data = pd.DataFrame({
            'age': [25, 30, 35],
            'income': [50000, 60000, 70000],
            'target': [0, 1, 0]
        })
        mock_read_csv.return_value = mock_data
        
        ingestion = DataIngestion()
        data = ingestion.load_reference_data()
        
        assert data is not None
        assert len(data) == 3
    
    @patch('src.data_ingestion.requests.get')
    def test_load_from_api(self, mock_get):
        """Test loading data from API."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {'id': 1, 'name': 'test1'},
            {'id': 2, 'name': 'test2'}
        ]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        ingestion = DataIngestion()
        data = ingestion._load_from_api('http://test.com/api')
        
        assert data is not None
        assert len(data) == 2

class TestAlertManager:
    """Test alert manager functionality."""
    
    @patch('src.alert_manager.smtplib.SMTP')
    def test_email_alert(self, mock_smtp):
        """Test email alert sending."""
        settings.EMAIL_SENDER = 'test@example.com'
        settings.EMAIL_PASSWORD = 'password'
        settings.EMAIL_RECIPIENTS = ['recipient@example.com']
        settings.SMTP_SERVER = 'smtp.example.com'
        settings.SMTP_PORT = 587
        
        alert = EmailAlert()
        result = alert.send(
            subject='Test Alert',
            body='This is a test'
        )
        
        assert result is True
    
    @patch('src.alert_manager.requests.post')
    def test_slack_alert(self, mock_post):
        """Test Slack alert sending."""
        settings.SLACK_WEBHOOK_URL = 'https://hooks.slack.com/test'
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        alert = SlackAlert()
        result = alert.send(
            text='Test Slack Alert'
        )
        
        assert result is True

class TestDriftDetector:
    """Test drift detector functionality."""
    
    def setup_method(self):
        """Setup test data."""
        self.reference_data = pd.DataFrame({
            'age': np.random.normal(35, 10, 1000),
            'income': np.random.normal(50000, 15000, 1000),
            'gender': np.random.choice(['M', 'F'], 1000),
            'occupation': np.random.choice(['Engineer', 'Teacher'], 1000),
            'target': np.random.choice([0, 1], 1000)
        })
        
        self.incoming_data = pd.DataFrame({
            'age': np.random.normal(45, 12, 1000),
            'income': np.random.normal(65000, 20000, 1000),
            'gender': np.random.choice(['M', 'F'], 1000, p=[0.6, 0.4]),
            'occupation': np.random.choice(['Engineer', 'Teacher'], 1000, p=[0.4, 0.6]),
            'target': np.random.choice([0, 1], 1000)
        })
    
    @patch('src.drift_detector.DataIngestion.load_reference_data')
    def test_drift_detection(self, mock_load):
        """Test drift detection functionality."""
        mock_load.return_value = self.reference_data
        
        detector = DriftDetector()
        results = detector.detect_drift(self.incoming_data)
        
        assert 'timestamp' in results
        assert 'features' in results
        assert 'overall_drift' in results
        assert 'drift_count' in results
        assert 'total_features' in results
