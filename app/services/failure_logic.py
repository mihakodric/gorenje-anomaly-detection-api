"""Failure detection logic based on temporal aggregation of predictions.

Implements configurable failure detection strategies:
1. Threshold-based: >= K anomalies in last N cycles
2. Consecutive: X consecutive anomalies in a row
"""

from typing import Any, Callable, Dict, List

from app.core.config import settings
from app.db.models import Prediction


COMPONENT_NAMES = ("heater", "pump", "motor")


class FailureDetectionService:
    """Service for detecting imminent failures based on historical patterns."""

    def __init__(
        self,
        window_size: int | None = None,
        threshold_count: int | None = None,
        consecutive_threshold: int | None = None,
        require_consecutive: bool | None = None,
    ) -> None:
        """Initialize failure detection with configurable thresholds."""
        self.window_size = window_size or settings.window_size
        self.threshold_count = threshold_count or settings.threshold_count
        self.consecutive_threshold = consecutive_threshold or settings.consecutive_threshold
        self.require_consecutive = (
            require_consecutive if require_consecutive is not None else settings.require_consecutive
        )

    def evaluate_failure(
        self,
        historical_predictions: List[Prediction],
    ) -> Dict[str, Any]:
        """Evaluate whether failure is imminent based on historical patterns."""
        total_records = len(historical_predictions)

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

        anomaly_count = sum(1 for pred in historical_predictions if pred.anomaly_detected)
        consecutive_anomalies = self._count_consecutive_matches(
            historical_predictions,
            lambda pred: pred.anomaly_detected,
        )
        failure_imminent = self._should_flag_failure(
            anomaly_count=anomaly_count,
            consecutive_anomalies=consecutive_anomalies,
        )

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

    def evaluate_component_failures(
        self,
        historical_predictions: List[Prediction],
    ) -> Dict[str, Dict[str, int | bool]]:
        """Evaluate failure risk for heater, pump, and motor independently."""
        results: Dict[str, Dict[str, int | bool]] = {}

        for component_name in COMPONENT_NAMES:
            anomaly_count = sum(
                1
                for pred in historical_predictions
                if component_name in (pred.failing_parts or [])
            )
            consecutive_anomalies = self._count_consecutive_matches(
                historical_predictions,
                lambda pred, component=component_name: component in (pred.failing_parts or []),
            )
            results[component_name] = {
                "failure_imminent": self._should_flag_failure(
                    anomaly_count=anomaly_count,
                    consecutive_anomalies=consecutive_anomalies,
                ),
                "anomaly_count": anomaly_count,
                "consecutive_anomalies": consecutive_anomalies,
            }

        return results

    def get_summary(self, historical_predictions: List[Prediction]) -> Dict[str, Any]:
        """Get a summary of historical predictions for debugging."""
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

    def _should_flag_failure(
        self,
        anomaly_count: int,
        consecutive_anomalies: int,
    ) -> bool:
        """Apply the configured failure strategy to anomaly counts."""
        if self.require_consecutive:
            return consecutive_anomalies >= self.consecutive_threshold
        return anomaly_count >= self.threshold_count

    @staticmethod
    def _count_consecutive_matches(
        historical_predictions: List[Prediction],
        predicate: Callable[[Prediction], bool],
    ) -> int:
        """Count consecutive matching records from the most recent prediction."""
        consecutive_matches = 0
        for pred in historical_predictions:
            if predicate(pred):
                consecutive_matches += 1
            else:
                break
        return consecutive_matches


def get_failure_detection_service() -> FailureDetectionService:
    """Get a failure detection service instance with default settings."""
    return FailureDetectionService()
