"""Basic tests for the API endpoints."""

import pytest
from httpx import AsyncClient
from datetime import datetime

from app.main import app


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
async def test_detect_anomaly_valid_request():
    """Test anomaly detection with a valid request."""
    request_data = {
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
        }
    }
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/detect_anomaly", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "anomaly_detected" in data
    assert "failure_imminent" in data
    assert "failing_parts" in data
    assert "auid" in data
    assert data["auid"] == request_data["auid"]
    assert isinstance(data["anomaly_detected"], bool)
    assert isinstance(data["failure_imminent"], bool)
    assert isinstance(data["failing_parts"], list)


@pytest.mark.asyncio
async def test_detect_anomaly_debug_mode():
    """Test anomaly detection with debug mode enabled."""
    request_data = {
        "auid": "test_auid_debug",
        "timestamp": "2026-03-31T11:00:00Z",
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
        }
    }
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/detect_anomaly?debug=true", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "history" in data
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
            # Missing other required fields
        },
        "data_102_65": {
            "diff_actuator1worktimeinseconds": 733,
        }
    }
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/detect_anomaly", json=request_data)
    
    # Should return 422 Validation Error
    assert response.status_code == 422
