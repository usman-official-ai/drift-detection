"""
Drift detection engine - Fixed for Evidently 0.7.21
Uses the correct import paths based on available modules
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import os

# ============================================
# EVIDENTLY IMPORTS - CORRECT FOR v0.7.21
# ============================================
from evidently import Report
from evidently import DataDefinition
from evidently import ColumnType

# DataDriftPreset is in the presets submodule
from evidently.presets import DataDriftPreset

from src.config import settings
from src.logger_config import get_logger
from src.data_ingestion import DataIngestion
from src.alert_manager import AlertManager
from src.exceptions import DriftDetectionError

logger = get_logger()

class StatisticalTests:
    """Statistical tests for drift detection."""
    
    @staticmethod
    def ks_test(sample1: pd.Series, sample2: pd.Series) -> Tuple[float, float]:
        """Perform Kolmogorov-Smirnov test."""
        sample1 = sample1.dropna()
        sample2 = sample2.dropna()
        if len(sample1) == 0 or len(sample2) == 0:
            return 0.0, 1.0
        try:
            statistic, p_value = stats.ks_2samp(sample1, sample2)
            return float(statistic), float(p_value)
        except Exception as e:
            logger.error(f"KS test failed: {str(e)}")
            return 0.0, 1.0
    
    @staticmethod
    def psi(expected: pd.Series, actual: pd.Series) -> float:
        """Calculate Population Stability Index."""
        expected = expected.dropna()
        actual = actual.dropna()
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
        
        try:
            is_numeric = pd.api.types.is_numeric_dtype(expected)
            
            if is_numeric:
                all_values = pd.concat([expected, actual])
                bins = min(10, int(np.sqrt(len(all_values))))
                bin_edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
                bin_edges = np.unique(bin_edges)
                
                expected_binned = pd.cut(expected, bins=bin_edges, include_lowest=True, duplicates="drop")
                actual_binned = pd.cut(actual, bins=bin_edges, include_lowest=True, duplicates="drop")
                
                expected_counts = expected_binned.value_counts(normalize=True)
                actual_counts = actual_binned.value_counts(normalize=True)
            else:
                expected_counts = expected.value_counts(normalize=True)
                actual_counts = actual.value_counts(normalize=True)
            
            all_categories = set(expected_counts.index) | set(actual_counts.index)
            psi = 0.0
            for category in all_categories:
                exp_pct = expected_counts.get(category, 0.0001)
                act_pct = actual_counts.get(category, 0.0001)
                if exp_pct == 0:
                    exp_pct = 0.0001
                if act_pct == 0:
                    act_pct = 0.0001
                psi += (act_pct - exp_pct) * np.log(act_pct / exp_pct)
            
            return float(psi)
        except Exception as e:
            logger.error(f"PSI calculation failed: {str(e)}")
            return 0.0

class DriftDetector:
    """Main drift detection engine."""
    
    def __init__(self):
        self.ingestion = DataIngestion()
        self.alert_manager = AlertManager()
        self.stats = StatisticalTests()
        self.reference_data = None
        self.incoming_data = None
        self.drift_results = {}
    
    def detect_drift(self, incoming_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Main method to detect drift in incoming data."""
        try:
            logger.info("Starting drift detection")
            
            # Load reference data
            self.reference_data = self.ingestion.load_reference_data()
            
            # Load incoming data
            if incoming_data is None:
                self.incoming_data = self.ingestion.load_incoming_data()
            else:
                self.incoming_data = incoming_data
            
            if self.incoming_data is None:
                raise DriftDetectionError("No incoming data available")
            
            # Perform drift detection
            self.drift_results = self._perform_drift_detection()
            
            # Generate Evidently report
            try:
                self._generate_evidently_report()
            except Exception as e:
                logger.warning(f"Evidently report skipped: {e}")
            
            # Send alerts if needed
            if self.drift_results.get("overall_drift", False):
                self.alert_manager.send_alerts(self.drift_results)
            
            logger.info(
                "Drift detection completed",
                extra={
                    "drift_count": self.drift_results.get("drift_count", 0),
                    "total_features": self.drift_results.get("total_features", 0),
                    "overall_drift": self.drift_results.get("overall_drift", False)
                }
            )
            
            return self.drift_results
            
        except Exception as e:
            logger.error(f"Drift detection failed: {str(e)}")
            raise DriftDetectionError(f"Drift detection failed: {str(e)}")
    
    def _perform_drift_detection(self) -> Dict[str, Any]:
        """Perform drift detection on all features."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "features": {},
            "drift_count": 0,
            "total_features": len(settings.NUMERICAL_FEATURES) + len(settings.CATEGORICAL_FEATURES),
            "overall_drift": False,
            "drift_percentage": 0.0,
            "reference_rows": len(self.reference_data) if self.reference_data is not None else 0,
            "incoming_rows": len(self.incoming_data) if self.incoming_data is not None else 0
        }
        
        # Numerical features with KS test
        for feature in settings.NUMERICAL_FEATURES:
            if feature in self.reference_data.columns and feature in self.incoming_data.columns:
                try:
                    ref_vals = self.reference_data[feature]
                    inc_vals = self.incoming_data[feature]
                    statistic, p_value = self.stats.ks_test(ref_vals, inc_vals)
                    drift_detected = p_value < settings.KS_THRESHOLD
                    
                    results["features"][feature] = {
                        "type": "numerical",
                        "test": "ks_test",
                        "statistic": statistic,
                        "p_value": p_value,
                        "drift_detected": drift_detected,
                        "threshold": settings.KS_THRESHOLD,
                        "reference_mean": float(ref_vals.mean()),
                        "incoming_mean": float(inc_vals.mean())
                    }
                    if drift_detected:
                        results["drift_count"] += 1
                except Exception as e:
                    logger.error(f"Error in KS test for {feature}: {e}")
                    results["features"][feature] = {"type": "numerical", "error": str(e), "drift_detected": False}
        
        # Categorical features with PSI
        for feature in settings.CATEGORICAL_FEATURES:
            if feature in self.reference_data.columns and feature in self.incoming_data.columns:
                try:
                    ref_vals = self.reference_data[feature].astype(str)
                    inc_vals = self.incoming_data[feature].astype(str)
                    psi_value = self.stats.psi(ref_vals, inc_vals)
                    drift_detected = psi_value > settings.PSI_THRESHOLD
                    
                    results["features"][feature] = {
                        "type": "categorical",
                        "test": "psi",
                        "psi": psi_value,
                        "drift_detected": drift_detected,
                        "threshold": settings.PSI_THRESHOLD,
                        "reference_unique": len(ref_vals.unique()),
                        "incoming_unique": len(inc_vals.unique())
                    }
                    if drift_detected:
                        results["drift_count"] += 1
                except Exception as e:
                    logger.error(f"Error in PSI for {feature}: {e}")
                    results["features"][feature] = {"type": "categorical", "error": str(e), "drift_detected": False}
        
        # Calculate overall drift
        results["drift_percentage"] = results["drift_count"] / max(results["total_features"], 1)
        results["overall_drift"] = results["drift_percentage"] > settings.DRIFT_PERCENTAGE_THRESHOLD
        
        return results
    
    def _generate_evidently_report(self):
        """Generate Evidently AI report."""
        try:
            # Create data definition
            data_definition = DataDefinition()
            
            # Add numerical columns
            for col in settings.NUMERICAL_FEATURES:
                if col in self.reference_data.columns:
                    data_definition.add_column(col, ColumnType.NUMERICAL)
            
            # Add categorical columns
            for col in settings.CATEGORICAL_FEATURES:
                if col in self.reference_data.columns:
                    data_definition.add_column(col, ColumnType.CATEGORICAL)
            
            # Add target if present
            if settings.TARGET_COLUMN in self.reference_data.columns:
                data_definition.add_column(settings.TARGET_COLUMN, ColumnType.TARGET)
            
            # Create and run report
            report = Report(metrics=[DataDriftPreset()])
            report.run(
                reference_data=self.reference_data,
                current_data=self.incoming_data,
                data_definition=data_definition
            )
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"{settings.REPORTS_PATH}/drift_report_{timestamp}.html"
            report.save_html(report_file)
            logger.info(f"Evidently report saved: {report_file}")
            
            # Also save JSON
            json_file = f"{settings.REPORTS_PATH}/drift_report_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(report.as_dict(), f, indent=2)
            logger.info(f"Evidently JSON report saved: {json_file}")
            
        except Exception as e:
            logger.warning(f"Evidently report generation skipped: {e}")
            raise  # Re-raise to see the error