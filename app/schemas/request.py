"""Request schemas for API endpoints."""

from typing import Any, Dict
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from utils.columns import region_0_cols, region_65_cols


class AnomalyDetectionRequest(BaseModel):
    """Request schema for POST /detect_anomaly endpoint.
    
    Validates top-level fields and ensures required fields from
    region_0_cols and region_65_cols are present in the data objects.
    """
    
    auid: str = Field(..., description="Appliance unique identifier")
    timestamp: str = Field(..., description="ISO8601 timestamp of the cycle")
    data_102_0: Dict[str, Any] = Field(..., description="Cycle settings and configuration")
    data_102_65: Dict[str, Any] = Field(..., description="Cycle measurement results")
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate that timestamp is a valid ISO8601 string."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid ISO8601 timestamp: {v}")
        return v
    
    @field_validator('data_102_0')
    @classmethod
    def validate_data_102_0(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that all required fields from region_0_cols are present."""
        missing_fields = [field for field in region_0_cols if field not in v]
        if missing_fields:
            raise ValueError(
                f"Missing required fields in data_102_0: {', '.join(missing_fields)}"
            )
        return v
    
    @field_validator('data_102_65')
    @classmethod
    def validate_data_102_65(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that all required fields from region_65_cols are present."""
        missing_fields = [field for field in region_65_cols if field not in v]
        if missing_fields:
            raise ValueError(
                f"Missing required fields in data_102_65: {', '.join(missing_fields)}"
            )
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "auid": "0000000000007442320000202400044930020",
                    "timestamp": "2026-03-01T10:15:00Z",
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
            ]
        }
    }
