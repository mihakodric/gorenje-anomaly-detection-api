"""Response schemas for API endpoints."""

from datetime import datetime
from typing import List, Literal

from pydantic import BaseModel, Field


class ComponentStatus(BaseModel):
    """Presentation status for a single washing machine component."""

    status: Literal["ok", "warning", "failing"] = Field(
        ...,
        description="Resolved component state for UI rendering",
    )
    color: str = Field(..., description="Hex color used to render the component in the SVG")


class ComponentStatusMap(BaseModel):
    """Resolved status and color for each tracked component."""

    heater: ComponentStatus
    pump: ComponentStatus
    motor: ComponentStatus


class AnomalyDetectionResponse(BaseModel):
    """Standard response schema for POST /detect_anomaly endpoint."""

    anomaly_detected: bool = Field(..., description="Whether current cycle shows an anomaly")
    failure_imminent: bool = Field(..., description="Whether failure is predicted based on historical patterns")
    failing_parts: List[str] = Field(..., description="List of failing component names (heater, pump, motor)")
    auid: str = Field(..., description="Appliance unique identifier")
    components: ComponentStatusMap = Field(..., description="Resolved per-component UI status and color")
    wm_svg: str = Field(..., description="Washing machine SVG text with component colors applied")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "anomaly_detected": True,
                    "failure_imminent": True,
                    "failing_parts": ["heater"],
                    "auid": "0000000000007442320000202400044930020",
                    "components": {
                        "heater": {"status": "failing", "color": "#CC0000"},
                        "pump": {"status": "ok", "color": "#669900"},
                        "motor": {"status": "ok", "color": "#669900"},
                    },
                    "wm_svg": "<?xml version='1.0' encoding='utf-8'?><svg>...</svg>",
                }
            ]
        }
    }


class ComponentPrediction(BaseModel):
    """Detailed prediction information for a single component."""

    predicted_value: float = Field(..., description="ML model predicted value")
    true_value: float = Field(..., description="Actual observed value")
    residual: float = Field(..., description="Absolute error between prediction and observation")
    sigma: float = Field(..., description="Residual standard deviation (1 sigma)")
    two_sigma: float = Field(..., description="Two standard deviations (2 sigma)")
    mean: float = Field(..., description="Mean residual")
    defined_limit: float = Field(..., description="Configured anomaly threshold")
    is_anomaly: bool = Field(..., description="Whether this component exceeds the threshold")


class PredictionDetails(BaseModel):
    """Detailed prediction breakdown per component."""

    heater: ComponentPrediction
    pump: ComponentPrediction
    motor: ComponentPrediction


class HistoricalSummary(BaseModel):
    """Summary of historical predictions used for failure detection."""

    total_records: int = Field(..., description="Total historical records retrieved")
    anomaly_count: int = Field(..., description="Number of anomalous cycles in history")
    consecutive_anomalies: int = Field(..., description="Length of current consecutive anomaly streak")
    window_size: int = Field(..., description="Configured window size for evaluation")
    threshold_count: int = Field(..., description="Configured threshold for anomaly count")
    consecutive_threshold: int = Field(..., description="Configured threshold for consecutive anomalies")
    require_consecutive: bool = Field(..., description="Whether consecutive mode is enabled")


class AnomalyDetectionDebugResponse(AnomalyDetectionResponse):
    """Extended response with debug information when ?debug=true is used."""

    predictions: PredictionDetails = Field(..., description="Detailed predictions for each component")
    history: HistoricalSummary = Field(..., description="Historical analysis summary")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "anomaly_detected": True,
                    "failure_imminent": True,
                    "failing_parts": ["heater"],
                    "auid": "0000000000007442320000202400044930020",
                    "components": {
                        "heater": {"status": "failing", "color": "#CC0000"},
                        "pump": {"status": "ok", "color": "#669900"},
                        "motor": {"status": "ok", "color": "#669900"},
                    },
                    "wm_svg": "<?xml version='1.0' encoding='utf-8'?><svg>...</svg>",
                    "predictions": {
                        "heater": {
                            "predicted_value": 950.5,
                            "true_value": 996.0,
                            "residual": 45.5,
                            "sigma": 15.2,
                            "two_sigma": 30.4,
                            "mean": 10.1,
                            "defined_limit": 40.5,
                            "is_anomaly": True,
                        },
                        "pump": {
                            "predicted_value": 730.0,
                            "true_value": 733.0,
                            "residual": 3.0,
                            "sigma": 20.5,
                            "two_sigma": 41.0,
                            "mean": 8.5,
                            "defined_limit": 49.5,
                            "is_anomaly": False,
                        },
                        "motor": {
                            "predicted_value": 8.5,
                            "true_value": 9.0,
                            "residual": 0.5,
                            "sigma": 5.2,
                            "two_sigma": 10.4,
                            "mean": 2.1,
                            "defined_limit": 12.5,
                            "is_anomaly": False,
                        },
                    },
                    "history": {
                        "total_records": 10,
                        "anomaly_count": 8,
                        "consecutive_anomalies": 3,
                        "window_size": 10,
                        "threshold_count": 8,
                        "consecutive_threshold": 3,
                        "require_consecutive": False,
                    },
                }
            ]
        }
    }


class AvailableAuidsResponse(BaseModel):
    """Response for listing AUIDs that have stored predictions."""

    auids: List[str] = Field(
        ...,
        description="Distinct appliance identifiers with at least one stored prediction",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "auids": [
                        "0000000000007442320000202400044930020",
                        "0000000000007442320000202400044930021",
                    ]
                }
            ]
        }
    }


class LatestPredictionResponse(BaseModel):
    """Response for the most recent stored prediction of one AUID."""

    id: int = Field(..., description="Stored prediction record identifier")
    auid: str = Field(..., description="Appliance unique identifier")
    timestamp: datetime = Field(..., description="Timestamp of the stored prediction")
    anomaly_detected: bool = Field(..., description="Whether the stored cycle was anomalous")
    failing_parts: List[str] = Field(..., description="Failing components detected for that cycle")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": 42,
                    "auid": "0000000000007442320000202400044930020",
                    "timestamp": "2026-03-31T10:15:00Z",
                    "anomaly_detected": True,
                    "failing_parts": ["heater"],
                }
            ]
        }
    }


class HealthCheckResponse(BaseModel):
    """Response for GET /health endpoint."""

    status: str = Field(default="healthy", description="API health status")
    version: str = Field(..., description="API version")
