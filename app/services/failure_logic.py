"""Failure detection logic based on temporal aggregation of predictions.

Implements configurable failure detection strategies:
1. Threshold-based: ≥ K anomalies in last N cycles
2. Consecutive: X consecutive anomalies in a row
"""

from typing import List, Dict, Any
from app.db.models import Prediction
from app.core.config import settings


class FailureDetectionService:
    """Service for detecting imminent failures based on historical patterns."""
    
    def __init__(
        self,
        window_size: int | None = None,
        threshold_count: int | None = None,
        consecutive_threshold: int | None = None,
        require_consecutive: bool | None = None,
    ) -> None:
        """Initialize failure detection with configurable thresholds.
        
        Args:
            window_size: Number of past cycles to consider.
            threshold_count: Minimum number of anomalies required.
            consecutive_threshold: Length of consecutive anomaly streak required.
            require_consecutive: Whether to require consecutive anomalies.
        """
        self.window_size = window_size or settings.window_size
        self.threshold_count = threshold_count or settings.threshold_count
        self.consecutive_threshold = consecutive_threshold or settings.consecutive_threshold
        self.require_consecutive = require_consecutive if require_consecutive is not None else settings.require_consecutive

    def evaluate_failure(
        self, 
        historical_predictions: List[Prediction]
    ) -> Dict[str, Any]:
        """Evaluate whether failure is imminent based on historical patterns.
        
        Args:
            historical_predictions: List of recent predictions, ordered by timestamp DESC.
            
        Returns:
            Dictionary containing:
                - failure_imminent: bool
                - total_records: int
                - anomaly_count: int
                - consecutive_anomalies: int
                - evaluation_details: Dict with thresholds and strategy
        """
        total_records = len(historical_predictions)
        
        # If no history, no failure
        if total_records == 0:
            return {
                "failure_imminent": False,
                "total_records": 0,
                "anomaly_count": 0,
                "consecutive_anomalies": 0,
                "window_size": self.window_size,
                "threshold_count": self.threshold_count,
                "consecutive_threshold": self.consecutive_threshold,
                "require_consecutive": self.require_consecutive,
            }
        
        # Count total anomalies in the retrieved records
        anomaly_count = sum(1 for pred in historical_predictions if pred.anomaly_detected)
        
        # Count consecutive anomalies from the most recent prediction
        consecutive_anomalies = 0
        for pred in historical_predictions:
            if pred.anomaly_detected:
                consecutive_anomalies += 1
            else:
                break  # Stop at first non-anomaly
        
        # Determine if failure is imminent based on strategy
        failure_imminent = False
        
        if self.require_consecutive:
            # Consecutive strategy: require X consecutive anomalies
            if consecutive_anomalies >= self.consecutive_threshold:
                failure_imminent = True
        else:
            # Threshold strategy: require ≥ K anomalies in last N cycles
            # Use available history even if < window_size
            if anomaly_count >= self.threshold_count:
                failure_imminent = True
        
        return {
            "failure_imminent": failure_imminent,
            "total_records": total_records,
            "anomaly_count": anomaly_count,
            "consecutive_anomalies": consecutive_anomalies,
            "window_size": self.window_size,
            "threshold_count": self.threshold_count,
            "consecutive_threshold": self.consecutive_threshold,
            "require_consecutive": self.require_consecutive,
        }

    def get_summary(self, historical_predictions: List[Prediction]) -> Dict[str, Any]:
        """Get a summary of historical predictions for debugging.
        
        Args:
            historical_predictions: List of recent predictions.
            
        Returns:
            Dictionary with summary statistics.
        """
        evaluation = self.evaluate_failure(historical_predictions)
        return {
            "total_records": evaluation["total_records"],
            "anomaly_count": evaluation["anomaly_count"],
            "consecutive_anomalies": evaluation["consecutive_anomalies"],
            "window_size": evaluation["window_size"],
            "threshold_count": evaluation["threshold_count"],
            "consecutive_threshold": evaluation["consecutive_threshold"],
            "require_consecutive": evaluation["require_consecutive"],
        }


def get_failure_detection_service() -> FailureDetectionService:
    """Get a failure detection service instance with default settings.
    
    Returns:
        FailureDetectionService configured with settings from environment.
    """
    return FailureDetectionService()
