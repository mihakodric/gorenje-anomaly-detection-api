"""API route handlers."""

from datetime import datetime
from typing import Any, Union

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app import __version__
from app.core.config import settings
from app.db import crud
from app.db.database import get_db
from app.schemas.request import AnomalyDetectionRequest
from app.schemas.response import (
    AvailableAuidsResponse,
    AnomalyDetectionDebugResponse,
    AnomalyDetectionResponse,
    ComponentPrediction,
    ComponentStatus,
    ComponentStatusMap,
    HealthCheckResponse,
    HistoricalSummary,
    LatestPredictionResponse,
    PredictionDetails,
)
from app.services.failure_logic import get_failure_detection_service
from app.services.inference import get_inference_service
from app.services.wm_svg import build_wm_svg


router = APIRouter()
COMPONENT_NAMES = ("heater", "pump", "motor")


@router.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check() -> HealthCheckResponse:
    """Health check endpoint for container orchestration."""
    return HealthCheckResponse(status="healthy", version=__version__)


@router.get(
    "/predictions/auids",
    response_model=AvailableAuidsResponse,
    tags=["Predictions"],
)
async def list_prediction_auids(
    db: AsyncSession = Depends(get_db),
) -> AvailableAuidsResponse:
    """List all AUIDs that have at least one stored prediction."""
    auids = await crud.get_all_auids_with_predictions(db=db)
    return AvailableAuidsResponse(auids=auids)


@router.get(
    "/predictions/{auid}/latest",
    response_model=LatestPredictionResponse,
    tags=["Predictions"],
)
async def get_latest_prediction(
    auid: str,
    db: AsyncSession = Depends(get_db),
) -> LatestPredictionResponse:
    """Return the most recent stored prediction for the provided AUID."""
    latest_prediction = await crud.get_latest_prediction_by_auid(db=db, auid=auid)
    if latest_prediction is None:
        raise HTTPException(
            status_code=404,
            detail=f"No stored predictions found for auid '{auid}'",
        )

    return LatestPredictionResponse(
        id=latest_prediction.id,
        auid=latest_prediction.auid,
        timestamp=latest_prediction.timestamp,
        anomaly_detected=latest_prediction.anomaly_detected,
        failing_parts=latest_prediction.failing_parts,
    )


@router.post(
    "/detect_anomaly",
    response_model=Union[AnomalyDetectionDebugResponse, AnomalyDetectionResponse],
    tags=["Anomaly Detection"],
)
async def detect_anomaly(
    request: AnomalyDetectionRequest,
    debug: bool = Query(False, description="Include detailed debug information in response"),
    db: AsyncSession = Depends(get_db),
) -> Union[AnomalyDetectionDebugResponse, AnomalyDetectionResponse]:
    """Detect anomalies in washing machine cycle data."""
    inference_service = get_inference_service()

    cycle_settings, cycle_result = inference_service.parse_input(
        request.data_102_0,
        request.data_102_65,
    )

    prediction_result = inference_service.predict(cycle_settings, cycle_result)
    timestamp = datetime.fromisoformat(request.timestamp.replace("Z", "+00:00"))

    await crud.create_prediction(
        db=db,
        auid=request.auid,
        timestamp=timestamp,
        anomaly_detected=prediction_result["anomaly_detected"],
        failing_parts=prediction_result["failing_parts"],
    )

    failure_service = get_failure_detection_service()
    historical_predictions = await crud.get_last_n_predictions(
        db=db,
        auid=request.auid,
        n=failure_service.window_size,
    )

    failure_evaluation = failure_service.evaluate_failure(historical_predictions)
    component_failure_evaluation = failure_service.evaluate_component_failures(historical_predictions)
    components = _build_component_status_map(
        prediction_result=prediction_result,
        component_failure_evaluation=component_failure_evaluation,
    )
    wm_svg = build_wm_svg(
        {
            component_name: getattr(components, component_name).color
            for component_name in COMPONENT_NAMES
        }
    )

    if debug:
        response = AnomalyDetectionDebugResponse(
            anomaly_detected=prediction_result["anomaly_detected"],
            failure_imminent=failure_evaluation["failure_imminent"],
            failing_parts=prediction_result["failing_parts"],
            auid=request.auid,
            components=components,
            wm_svg=wm_svg,
            predictions=PredictionDetails(
                heater=ComponentPrediction(**prediction_result["predictions"]["heater"]),
                pump=ComponentPrediction(**prediction_result["predictions"]["pump"]),
                motor=ComponentPrediction(**prediction_result["predictions"]["motor"]),
            ),
            history=HistoricalSummary(
                total_records=failure_evaluation["total_records"],
                anomaly_count=failure_evaluation["anomaly_count"],
                consecutive_anomalies=failure_evaluation["consecutive_anomalies"],
                window_size=failure_evaluation["window_size"],
                threshold_count=failure_evaluation["threshold_count"],
                consecutive_threshold=failure_evaluation["consecutive_threshold"],
                require_consecutive=failure_evaluation["require_consecutive"],
            ),
        )
    else:
        response = AnomalyDetectionResponse(
            anomaly_detected=prediction_result["anomaly_detected"],
            failure_imminent=failure_evaluation["failure_imminent"],
            failing_parts=prediction_result["failing_parts"],
            auid=request.auid,
            components=components,
            wm_svg=wm_svg,
        )

    return response


def _build_component_status_map(
    prediction_result: dict[str, Any],
    component_failure_evaluation: dict[str, dict[str, int | bool]],
) -> ComponentStatusMap:
    """Resolve component presentation states from current and historical signals."""
    return ComponentStatusMap(
        heater=_build_component_status(
            component_name="heater",
            prediction_result=prediction_result,
            component_failure_evaluation=component_failure_evaluation,
        ),
        pump=_build_component_status(
            component_name="pump",
            prediction_result=prediction_result,
            component_failure_evaluation=component_failure_evaluation,
        ),
        motor=_build_component_status(
            component_name="motor",
            prediction_result=prediction_result,
            component_failure_evaluation=component_failure_evaluation,
        ),
    )


def _build_component_status(
    component_name: str,
    prediction_result: dict[str, Any],
    component_failure_evaluation: dict[str, dict[str, int | bool]],
) -> ComponentStatus:
    """Resolve one component to ok, warning, or failing for the response."""
    is_failing = bool(component_failure_evaluation[component_name]["failure_imminent"])
    is_anomaly = bool(prediction_result["predictions"][component_name]["is_anomaly"])

    if is_failing:
        return ComponentStatus(status="failing", color=settings.component_failing_color)
    if is_anomaly:
        return ComponentStatus(status="warning", color=settings.component_warning_color)
    return ComponentStatus(status="ok", color=settings.component_ok_color)
