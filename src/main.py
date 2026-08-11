 # src/main.py
"""
Main entry point for the drift detection system.
Supports both one-time execution and scheduled runs.
"""

import sys
import argparse
import signal
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from src.config import settings
from src.logger_config import get_logger
from src.drift_detector import DriftDetector
from src.data_ingestion import DataIngestion
from src.exceptions import DriftDetectionError

logger = get_logger()

class DriftDetectionPipeline:
    """
    Main pipeline orchestrator for drift detection.
    """
    
    def __init__(self):
        self.detector = DriftDetector()
        self.ingestion = DataIngestion()
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def run_once(
        self,
        source: str = "file",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run drift detection once.
        
        Args:
            source: Data source type
            **kwargs: Additional arguments for data loading
        
        Returns:
            Drift detection results
        """
        logger.info("Starting one-time drift detection run")
        
        try:
            # Validate configuration
            config_errors = settings.validate_configuration()
            if config_errors:
                logger.error(f"Configuration errors: {config_errors}")
                return {"status": "error", "errors": config_errors}
            
            # Load incoming data
            incoming_data = self.ingestion.load_incoming_data(source, **kwargs)
            
            # Run drift detection
            results = self.detector.detect_drift(incoming_data)
            
            # Log summary
            logger.info(
                "Drift detection completed",
                extra={
                    "drift_count": results["drift_count"],
                    "drift_percentage": results["drift_percentage"],
                    "overall_drift": results["overall_drift"]
                }
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def run_scheduled(self):
        """
        Run drift detection on a schedule.
        """
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
        
        logger.info(
            "Starting scheduled drift detection",
            extra={
                "interval_hours": settings.SCHEDULE_INTERVAL_HOURS,
                "start_time": settings.SCHEDULE_START_TIME
            }
        )
        
        scheduler = BlockingScheduler()
        
        # Parse start time
        hour, minute = map(int, settings.SCHEDULE_START_TIME.split(':'))
        
        # Schedule job
        scheduler.add_job(
            self.run_once,
            trigger=CronTrigger(hour=hour, minute=minute),
            args=["file"],
            id="drift_detection_job",
            name="Drift Detection Job",
            max_instances=1,
            misfire_grace_time=3600
        )
        
        # Also run immediately on start
        self.run_once()
        
        try:
            scheduler.start()
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            scheduler.shutdown()
    
    def run_continuous(self, interval_seconds: int = 3600):
        """
        Run drift detection continuously at specified interval.
        
        Args:
            interval_seconds: Interval between runs in seconds
        """
        logger.info(f"Starting continuous mode with {interval_seconds}s interval")
        
        while self.running:
            try:
                self.run_once()
                
                # Sleep until next run
                if self.running:
                    time.sleep(interval_seconds)
                    
            except Exception as e:
                logger.error(f"Continuous run error: {str(e)}")
                if self.running:
                    time.sleep(60)  # Wait before retrying

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Data Drift Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run once with file data
  python main.py --mode once --source file
  
  # Run once with API data
  python main.py --mode once --source api --api-url http://api.example.com/data
  
  # Run on schedule
  python main.py --mode schedule
  
  # Run continuously
  python main.py --mode continuous --interval 3600
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["once", "schedule", "continuous"],
        default="once",
        help="Execution mode"
    )
    
    parser.add_argument(
        "--source",
        choices=["file", "api", "database"],
        default="file",
        help="Data source type"
    )
    
    parser.add_argument(
        "--api-url",
        type=str,
        help="API URL for data source (when --source is 'api')"
    )
    
    parser.add_argument(
        "--file",
        type=str,
        help="File path for data source (when --source is 'file')"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Interval in seconds for continuous mode"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create pipeline
    pipeline = DriftDetectionPipeline()
    
    # Run based on mode
    if args.mode == "once":
        kwargs = {}
        if args.api_url:
            kwargs["api_url"] = args.api_url
        if args.file:
            kwargs["file_path"] = args.file
        
        results = pipeline.run_once(args.source, **kwargs)
        
        # Print summary
        if results.get("status") == "error":
            print(f"\nERROR: {results.get('message', 'Unknown error')}")
            sys.exit(1)
        else:
            print("\n" + "=" * 60)
            print("DRIFT DETECTION SUMMARY")
            print("=" * 60)
            print(f"Timestamp: {results.get('timestamp', 'N/A')}")
            print(f"Features Analyzed: {results.get('total_features', 0)}")
            print(f"Features with Drift: {results.get('drift_count', 0)}")
            print(f"Drift Percentage: {results.get('drift_percentage', 0):.1%}")
            print(f"Overall Drift: {'YES' if results.get('overall_drift', False) else 'NO'}")
            
            if results.get("drift_count", 0) > 0:
                print("\nDrifting Features:")
                for feature, details in results.get("features", {}).items():
                    if details.get("drift_detected", False):
                        if details.get("type") == "numerical":
                            print(f"  • {feature}: p-value = {details.get('p_value', 0):.4f}")
                        else:
                            print(f"  • {feature}: PSI = {details.get('psi', 0):.4f}")
            
            print("=" * 60)
    
    elif args.mode == "schedule":
        pipeline.run_scheduled()
    
    elif args.mode == "continuous":
        pipeline.run_continuous(args.interval)

if __name__ == "__main__":
    main()
