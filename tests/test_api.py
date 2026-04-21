"""Basic tests for the API endpoints."""

from datetime import datetime, timezone
from typing import Any

import pytest
from httpx import AsyncClient

import app.api.routes as routes
from app.db.crud import create_prediction
from app.db.database import AsyncSessionLocal
from app.main import app


VALID_REQUEST_DATA = {
    "auid": "0000000000007442320000202400044930020",
    "timestamp": "2026-03-31T10:15:00Z",
    "data_102_0": {
        "status": 6,
        "error_3": 1,
        "error_6": 1,
        "error_7": 1,
        "error_7_1": 1,
        "error_8": 1,
        "error_12_1": 1,
        "error_12_10": 2,
        "alarm_1": 1,
        "alarm_2": 1,
        "alarm_3": 1,
        "alarm_16": 1,
        "extra_rinse_status": 0,
        "power_save_status": 0,
        "selected_program_duration_in_minutes": 59,
        "selected_program_id_status": 7,
        "selected_program_load_status": 0,
        "selected_program_mode_status": 1,
        "selected_program_mode2_status": 0,
        "selected_program_prewash_status": 1,
        "selected_program_set_temperature_status": 4,
        "selected_program_small_load_status": 0,
        "selected_program_washing_spin_speed_rpm_status": 8,
        "selected_program_water_pluse_status": 1,
        "stain_program_set_clothes_type_status": 0,
        "stain_program_set_stain_status": 0,
        "time_program_set_time_status": 0,
        "time_save_status": 0,
    },
    "data_102_65": {
        "diff_actuator1worktimeinseconds": 733,
        "diff_actuator13worktimeinseconds": 996,
        "diff_totalmotorenergyconsumtion": 9,
        "diff_loadweightedcycles": 729,
        "diff_cumulativeeccentricload": 65,
    },
}


class FakeInferenceService:
    """Simple inference stub for route tests."""

    def __init__(self, prediction_result: dict[str, Any]) -> None:
        self.prediction_result = prediction_result

    def parse_input(self, cycle_settings_raw: dict[str, Any], cycle_result_raw: dict[str, Any]):
        return cycle_settings_raw, cycle_result_raw

    def predict(self, cycle_settings: dict[str, Any], cycle_result: dict[str, Any]) -> dict[str, Any]:
        return self.prediction_result


class FakeFailureService:
    """Simple failure evaluation stub for route tests."""

    def __init__(
        self,
        failure_imminent: bool,
        component_failures: dict[str, bool],
    ) -> None:
        self.window_size = 10
        self.threshold_count = 8
        self.consecutive_threshold = 3
        self.require_consecutive = False
        self.failure_imminent = failure_imminent
        self.component_failures = component_failures

    def evaluate_failure(self, historical_predictions: list[Any]) -> dict[str, Any]:
        return {
            "failure_imminent": self.failure_imminent,
            "total_records": len(historical_predictions),
            "anomaly_count": 0,
            "consecutive_anomalies": 0,
            "window_size": self.window_size,
            "threshold_count": self.threshold_count,
            "consecutive_threshold": self.consecutive_threshold,
            "require_consecutive": self.require_consecutive,
        }

    def evaluate_component_failures(self, historical_predictions: list[Any]) -> dict[str, dict[str, int | bool]]:
        return {
            component_name: {
                "failure_imminent": is_failing,
                "anomaly_count": int(is_failing),
                "consecutive_anomalies": int(is_failing),
            }
            for component_name, is_failing in self.component_failures.items()
        }


def build_prediction_result(
    heater_anomaly: bool = False,
    pump_anomaly: bool = False,
    motor_anomaly: bool = False,
) -> dict[str, Any]:
    """Build a deterministic prediction payload for route tests."""
    component_anomalies = {
        "heater": heater_anomaly,
        "pump": pump_anomaly,
        "motor": motor_anomaly,
    }
    failing_parts = [
        component_name for component_name, is_anomaly in component_anomalies.items() if is_anomaly
    ]

    return {
        "anomaly_detected": any(component_anomalies.values()),
        "failing_parts": failing_parts,
        "predictions": {
            component_name: {
                "predicted_value": 100.0,
                "true_value": 100.0 if not is_anomaly else 150.0,
                "residual": 0.0 if not is_anomaly else 50.0,
                "sigma": 10.0,
                "two_sigma": 20.0,
                "mean": 5.0,
                "defined_limit": 30.0,
                "is_anomaly": is_anomaly,
            }
            for component_name, is_anomaly in component_anomalies.items()
        },
    }


def patch_route_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    prediction_result: dict[str, Any],
    failure_imminent: bool,
    component_failures: dict[str, bool],
) -> None:
    """Patch route dependencies so response tests stay deterministic."""

    async def fake_create_prediction(**kwargs):
        return None

    async def fake_get_last_n_predictions(**kwargs):
        return [object(), object()]

    monkeypatch.setattr(routes, "get_inference_service", lambda: FakeInferenceService(prediction_result))
    monkeypatch.setattr(
        routes,
        "get_failure_detection_service",
        lambda: FakeFailureService(
            failure_imminent=failure_imminent,
            component_failures=component_failures,
        ),
    )
    monkeypatch.setattr(routes.crud, "create_prediction", fake_create_prediction)
    monkeypatch.setattr(routes.crud, "get_last_n_predictions", fake_get_last_n_predictions)


@pytest.mark.asyncio
async def test_health_check():
    """Test the health check endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


@pytest.mark.asyncio
async def test_root_endpoint():
    """Test the root endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_list_prediction_auids():
    """Test listing distinct AUIDs with stored predictions."""
    async with AsyncSessionLocal() as session:
        await create_prediction(
            db=session,
            auid="auid-b",
            timestamp=datetime(2026, 3, 31, 10, 15, tzinfo=timezone.utc),
            anomaly_detected=False,
            failing_parts=[],
        )
        await create_prediction(
            db=session,
            auid="auid-a",
            timestamp=datetime(2026, 3, 31, 11, 15, tzinfo=timezone.utc),
            anomaly_detected=True,
            failing_parts=["heater"],
        )
        await create_prediction(
            db=session,
            auid="auid-a",
            timestamp=datetime(2026, 3, 31, 12, 15, tzinfo=timezone.utc),
            anomaly_detected=False,
            failing_parts=[],
        )

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/predictions/auids")

    assert response.status_code == 200
    assert response.json() == {"auids": ["auid-a", "auid-b"]}


@pytest.mark.asyncio
async def test_get_latest_prediction_by_auid():
    """Test retrieving the newest stored prediction for one AUID."""
    async with AsyncSessionLocal() as session:
        await create_prediction(
            db=session,
            auid="target-auid",
            timestamp=datetime(2026, 3, 31, 10, 15, tzinfo=timezone.utc),
            anomaly_detected=False,
            failing_parts=[],
        )
        await create_prediction(
            db=session,
            auid="target-auid",
            timestamp=datetime(2026, 3, 31, 11, 15, tzinfo=timezone.utc),
            anomaly_detected=True,
            failing_parts=["pump"],
        )
        await create_prediction(
            db=session,
            auid="other-auid",
            timestamp=datetime(2026, 3, 31, 12, 15, tzinfo=timezone.utc),
            anomaly_detected=True,
            failing_parts=["motor"],
        )

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/predictions/target-auid/latest")

    assert response.status_code == 200
    data = response.json()
    assert data["auid"] == "target-auid"
    assert data["anomaly_detected"] is True
    assert data["failing_parts"] == ["pump"]
    assert data["timestamp"] == "2026-03-31T11:15:00Z"


@pytest.mark.asyncio
async def test_get_latest_prediction_by_auid_not_found():
    """Test retrieving latest prediction for an unknown AUID."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/predictions/missing-auid/latest")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "No stored predictions found for auid 'missing-auid'"
    }


@pytest.mark.asyncio
async def test_detect_anomaly_all_green_response(monkeypatch: pytest.MonkeyPatch):
    """Test anomaly detection response when all components are healthy."""
    patch_route_dependencies(
        monkeypatch=monkeypatch,
        prediction_result=build_prediction_result(),
        failure_imminent=False,
        component_failures={"heater": False, "pump": False, "motor": False},
    )

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/detect_anomaly", json=VALID_REQUEST_DATA)

    assert response.status_code == 200
    data = response.json()
    assert data["anomaly_detected"] is False
    assert data["failure_imminent"] is False
    assert data["failing_parts"] == []
    assert data["components"]["heater"] == {"status": "ok", "color": "#669900"}
    assert data["components"]["pump"] == {"status": "ok", "color": "#669900"}
    assert data["components"]["motor"] == {"status": "ok", "color": "#669900"}
    assert "wm_svg" in data
    assert "Heater" in data["wm_svg"]
    assert "#669900" in data["wm_svg"]


@pytest.mark.asyncio
async def test_detect_anomaly_warning_component_is_yellow(monkeypatch: pytest.MonkeyPatch):
    """Test that an anomalous component becomes yellow when not yet failing."""
    patch_route_dependencies(
        monkeypatch=monkeypatch,
        prediction_result=build_prediction_result(motor_anomaly=True),
        failure_imminent=False,
        component_failures={"heater": False, "pump": False, "motor": False},
    )

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/detect_anomaly", json=VALID_REQUEST_DATA)

    assert response.status_code == 200
    data = response.json()
    assert data["components"]["motor"] == {"status": "warning", "color": "#FFCC00"}
    assert data["components"]["heater"] == {"status": "ok", "color": "#669900"}
    assert "#FFCC00" in data["wm_svg"]


@pytest.mark.asyncio
async def test_detect_anomaly_failing_component_is_red(monkeypatch: pytest.MonkeyPatch):
    """Test that a historically failing component becomes red."""
    patch_route_dependencies(
        monkeypatch=monkeypatch,
        prediction_result=build_prediction_result(),
        failure_imminent=True,
        component_failures={"heater": True, "pump": False, "motor": False},
    )

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/detect_anomaly", json=VALID_REQUEST_DATA)

    assert response.status_code == 200
    data = response.json()
    assert data["failure_imminent"] is True
    assert data["components"]["heater"] == {"status": "failing", "color": "#CC0000"}
    assert data["components"]["pump"] == {"status": "ok", "color": "#669900"}
    assert "#CC0000" in data["wm_svg"]


@pytest.mark.asyncio
async def test_detect_anomaly_debug_mode(monkeypatch: pytest.MonkeyPatch):
    """Test anomaly detection with debug mode enabled."""
    patch_route_dependencies(
        monkeypatch=monkeypatch,
        prediction_result=build_prediction_result(heater_anomaly=True),
        failure_imminent=False,
        component_failures={"heater": False, "pump": False, "motor": False},
    )

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/detect_anomaly?debug=true", json=VALID_REQUEST_DATA)

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "history" in data
    assert "components" in data
    assert "wm_svg" in data
    assert data["components"]["heater"] == {"status": "warning", "color": "#FFCC00"}
    assert "heater" in data["predictions"]
    assert "pump" in data["predictions"]
    assert "motor" in data["predictions"]


@pytest.mark.asyncio
async def test_detect_anomaly_missing_fields():
    """Test anomaly detection with missing required fields."""
    request_data = {
        "auid": "test_missing_fields",
        "timestamp": "2026-03-31T12:00:00Z",
        "data_102_0": {
            "status": 6,
        },
        "data_102_65": {
            "diff_actuator1worktimeinseconds": 733,
        },
    }

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/detect_anomaly", json=request_data)

    assert response.status_code == 422
