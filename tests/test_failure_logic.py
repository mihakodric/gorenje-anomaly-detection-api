"""Tests for failure detection logic."""

import pytest
from datetime import datetime, timedelta

from app.services.failure_logic import FailureDetectionService
from app.db.models import Prediction


def create_mock_prediction(
    auid: str,
    timestamp: datetime,
    anomaly_detected: bool,
    failing_parts: list = None
) -> Prediction:
    """Create a mock Prediction object for testing."""
    pred = Prediction(
        auid=auid,
        timestamp=timestamp,
        anomaly_detected=anomaly_detected,
        failing_parts=failing_parts or [],
    )
    return pred


def test_threshold_strategy_triggers_failure():
    """Test that threshold strategy detects failure when >= K anomalies."""
    service = FailureDetectionService(
        window_size=10,
        threshold_count=8,
        consecutive_threshold=3,
        require_consecutive=False,
    )
    
    # Create 10 predictions with 8 anomalies
    base_time = datetime.now()
    predictions = []
    for i in range(10):
        # First 8 are anomalies, last 2 are normal
        is_anomaly = i < 8
        pred = create_mock_prediction(
            auid="test_auid",
            timestamp=base_time - timedelta(hours=9-i),
            anomaly_detected=is_anomaly,
        )
        predictions.append(pred)
    
    # Reverse to simulate DESC order from DB
    predictions.reverse()
    
    result = service.evaluate_failure(predictions)
    
    assert result["failure_imminent"] is True
    assert result["anomaly_count"] == 8
    assert result["total_records"] == 10


def test_threshold_strategy_no_failure():
    """Test that threshold strategy doesn't detect failure when < K anomalies."""
    service = FailureDetectionService(
        window_size=10,
        threshold_count=8,
        consecutive_threshold=3,
        require_consecutive=False,
    )
    
    # Create 10 predictions with only 5 anomalies
    base_time = datetime.now()
    predictions = []
    for i in range(10):
        is_anomaly = i < 5
        pred = create_mock_prediction(
            auid="test_auid",
            timestamp=base_time - timedelta(hours=9-i),
            anomaly_detected=is_anomaly,
        )
        predictions.append(pred)
    
    predictions.reverse()
    
    result = service.evaluate_failure(predictions)
    
    assert result["failure_imminent"] is False
    assert result["anomaly_count"] == 5


def test_consecutive_strategy_triggers_failure():
    """Test that consecutive strategy detects failure with X consecutive anomalies."""
    service = FailureDetectionService(
        window_size=10,
        threshold_count=8,
        consecutive_threshold=3,
        require_consecutive=True,
    )
    
    # Create predictions with 3 consecutive anomalies in the most recent records
    base_time = datetime.now()
    predictions = []
    for i in range(10):
        is_anomaly = i >= 7
        pred = create_mock_prediction(
            auid="test_auid",
            timestamp=base_time - timedelta(hours=9-i),
            anomaly_detected=is_anomaly,
        )
        predictions.append(pred)
    
    predictions.reverse()
    
    result = service.evaluate_failure(predictions)
    
    assert result["failure_imminent"] is True
    assert result["consecutive_anomalies"] == 3


def test_consecutive_strategy_no_failure():
    """Test that consecutive strategy doesn't detect failure without enough consecutive anomalies."""
    service = FailureDetectionService(
        window_size=10,
        threshold_count=8,
        consecutive_threshold=3,
        require_consecutive=True,
    )
    
    # Create predictions with anomalies but not consecutive
    base_time = datetime.now()
    predictions = []
    for i in range(10):
        # Alternating pattern with the most recent record as an anomaly
        is_anomaly = i % 2 == 1
        pred = create_mock_prediction(
            auid="test_auid",
            timestamp=base_time - timedelta(hours=9-i),
            anomaly_detected=is_anomaly,
        )
        predictions.append(pred)
    
    predictions.reverse()
    
    result = service.evaluate_failure(predictions)
    
    assert result["failure_imminent"] is False
    assert result["consecutive_anomalies"] == 1  # Only 1 consecutive (most recent)


def test_sparse_history():
    """Test behavior when history has fewer records than window size."""
    service = FailureDetectionService(
        window_size=10,
        threshold_count=8,
        consecutive_threshold=3,
        require_consecutive=False,
    )
    
    # Only 3 predictions, all anomalies
    base_time = datetime.now()
    predictions = [
        create_mock_prediction("test", base_time - timedelta(hours=2), True),
        create_mock_prediction("test", base_time - timedelta(hours=1), True),
        create_mock_prediction("test", base_time, True),
    ]
    
    predictions.reverse()
    
    result = service.evaluate_failure(predictions)
    
    # Should use available history (3 anomalies < 8 threshold)
    assert result["failure_imminent"] is False
    assert result["total_records"] == 3
    assert result["anomaly_count"] == 3


def test_empty_history():
    """Test behavior with no historical data."""
    service = FailureDetectionService(
        window_size=10,
        threshold_count=8,
        consecutive_threshold=3,
        require_consecutive=False,
    )
    
    result = service.evaluate_failure([])
    
    assert result["failure_imminent"] is False
    assert result["total_records"] == 0
    assert result["anomaly_count"] == 0
    assert result["consecutive_anomalies"] == 0
