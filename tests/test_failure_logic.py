"""Tests for failure detection logic."""

from datetime import datetime, timedelta

from app.db.models import Prediction
from app.services.failure_logic import FailureDetectionService


def create_mock_prediction(
    auid: str,
    timestamp: datetime,
    anomaly_detected: bool,
    failing_parts: list | None = None,
) -> Prediction:
    """Create a mock Prediction object for testing."""
    return Prediction(
        auid=auid,
        timestamp=timestamp,
        anomaly_detected=anomaly_detected,
        failing_parts=failing_parts or [],
    )


def test_threshold_strategy_triggers_failure():
    """Test that threshold strategy detects failure when >= K anomalies."""
    service = FailureDetectionService(
        window_size=10,
        threshold_count=8,
        consecutive_threshold=3,
        require_consecutive=False,
    )

    base_time = datetime.now()
    predictions = []
    for i in range(10):
        predictions.append(
            create_mock_prediction(
                auid="test_auid",
                timestamp=base_time - timedelta(hours=9 - i),
                anomaly_detected=i < 8,
            )
        )

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

    base_time = datetime.now()
    predictions = []
    for i in range(10):
        predictions.append(
            create_mock_prediction(
                auid="test_auid",
                timestamp=base_time - timedelta(hours=9 - i),
                anomaly_detected=i < 5,
            )
        )

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

    base_time = datetime.now()
    predictions = []
    for i in range(10):
        predictions.append(
            create_mock_prediction(
                auid="test_auid",
                timestamp=base_time - timedelta(hours=9 - i),
                anomaly_detected=i >= 7,
            )
        )

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

    base_time = datetime.now()
    predictions = []
    for i in range(10):
        predictions.append(
            create_mock_prediction(
                auid="test_auid",
                timestamp=base_time - timedelta(hours=9 - i),
                anomaly_detected=i % 2 == 1,
            )
        )

    predictions.reverse()
    result = service.evaluate_failure(predictions)

    assert result["failure_imminent"] is False
    assert result["consecutive_anomalies"] == 1


def test_component_threshold_strategy_triggers_failure():
    """Test threshold-based failure evaluation for a single component."""
    service = FailureDetectionService(
        window_size=10,
        threshold_count=3,
        consecutive_threshold=2,
        require_consecutive=False,
    )

    base_time = datetime.now()
    predictions = [
        create_mock_prediction("test", base_time - timedelta(hours=3), True, ["heater"]),
        create_mock_prediction("test", base_time - timedelta(hours=2), True, ["heater"]),
        create_mock_prediction("test", base_time - timedelta(hours=1), True, ["heater"]),
        create_mock_prediction("test", base_time, False, []),
    ]

    predictions.reverse()
    result = service.evaluate_component_failures(predictions)

    assert result["heater"]["failure_imminent"] is True
    assert result["pump"]["failure_imminent"] is False
    assert result["motor"]["failure_imminent"] is False


def test_component_consecutive_strategy_triggers_failure():
    """Test consecutive failure evaluation for a single component."""
    service = FailureDetectionService(
        window_size=10,
        threshold_count=8,
        consecutive_threshold=2,
        require_consecutive=True,
    )

    base_time = datetime.now()
    predictions = [
        create_mock_prediction("test", base_time - timedelta(hours=2), False, []),
        create_mock_prediction("test", base_time - timedelta(hours=1), True, ["pump"]),
        create_mock_prediction("test", base_time, True, ["pump"]),
    ]

    predictions.reverse()
    result = service.evaluate_component_failures(predictions)

    assert result["pump"]["failure_imminent"] is True
    assert result["pump"]["consecutive_anomalies"] == 2


def test_component_mixed_history_keeps_red_and_yellow_separate():
    """Test mixed component history where one component can fail while another does not."""
    service = FailureDetectionService(
        window_size=10,
        threshold_count=2,
        consecutive_threshold=2,
        require_consecutive=False,
    )

    base_time = datetime.now()
    predictions = [
        create_mock_prediction("test", base_time - timedelta(hours=2), True, ["heater"]),
        create_mock_prediction("test", base_time - timedelta(hours=1), True, ["heater"]),
        create_mock_prediction("test", base_time, True, ["motor"]),
    ]

    predictions.reverse()
    result = service.evaluate_component_failures(predictions)

    assert result["heater"]["failure_imminent"] is True
    assert result["motor"]["failure_imminent"] is False
    assert result["pump"]["failure_imminent"] is False


def test_sparse_history():
    """Test behavior when history has fewer records than window size."""
    service = FailureDetectionService(
        window_size=10,
        threshold_count=8,
        consecutive_threshold=3,
        require_consecutive=False,
    )

    base_time = datetime.now()
    predictions = [
        create_mock_prediction("test", base_time - timedelta(hours=2), True, ["heater"]),
        create_mock_prediction("test", base_time - timedelta(hours=1), True, ["pump"]),
        create_mock_prediction("test", base_time, True, ["motor"]),
    ]

    predictions.reverse()
    result = service.evaluate_failure(predictions)
    component_result = service.evaluate_component_failures(predictions)

    assert result["failure_imminent"] is False
    assert result["total_records"] == 3
    assert result["anomaly_count"] == 3
    assert component_result["heater"]["failure_imminent"] is False
    assert component_result["pump"]["failure_imminent"] is False
    assert component_result["motor"]["failure_imminent"] is False


def test_empty_history():
    """Test behavior with no historical data."""
    service = FailureDetectionService(
        window_size=10,
        threshold_count=8,
        consecutive_threshold=3,
        require_consecutive=False,
    )

    result = service.evaluate_failure([])
    component_result = service.evaluate_component_failures([])

    assert result["failure_imminent"] is False
    assert result["total_records"] == 0
    assert result["anomaly_count"] == 0
    assert result["consecutive_anomalies"] == 0
    assert component_result["heater"]["failure_imminent"] is False
    assert component_result["pump"]["failure_imminent"] is False
    assert component_result["motor"]["failure_imminent"] is False
