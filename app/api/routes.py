"""API route handlers."""

from datetime import datetime
from typing import Union

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db import crud
from app.schemas.request import AnomalyDetectionRequest
from app.schemas.response import (
    AnomalyDetectionResponse,
    AnomalyDetectionDebugResponse,
    HealthCheckResponse,
    ComponentPrediction,
    PredictionDetails,
    HistoricalSummary,
)
from app.services.inference import get_inference_service
from app.services.failure_logic import get_failure_detection_service
from app import __version__


router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check() -> HealthCheckResponse:
    """Health check endpoint for container orchestration.
    
    Returns:
        Health status and API version.
    """
    return HealthCheckResponse(status="healthy", version=__version__)


@router.post(
    "/detect_anomaly",
    response_model=Union[AnomalyDetectionResponse, AnomalyDetectionDebugResponse],
    tags=["Anomaly Detection"],
)
async def detect_anomaly(
    request: AnomalyDetectionRequest,
    debug: bool = Query(False, description="Include detailed debug information in response"),
    db: AsyncSession = Depends(get_db),
) -> Union[AnomalyDetectionResponse, AnomalyDetectionDebugResponse]:
    """Detect anomalies in washing machine cycle data.
    
    This endpoint:
    1. Runs ML inference on the provided cycle data
    2. Stores the prediction in the database
    3. Retrieves historical predictions for the AUID
    4. Evaluates failure risk based on temporal patterns
    
    Args:
        request: Cycle data including settings and results.
        debug: If True, returns detailed prediction and historical information.
        db: Database session (injected).
        
    Returns:
        AnomalyDetectionResponse or AnomalyDetectionDebugResponse based on debug flag.
    """
    # Get inference service
    inference_service = get_inference_service()
    
    # Parse input data
    cycle_settings, cycle_result = inference_service.parse_input(
        request.data_102_0,
        request.data_102_65,
    )
    
    # Run ML inference
    prediction_result = inference_service.predict(cycle_settings, cycle_result)
    
    # Parse timestamp
    timestamp = datetime.fromisoformat(request.timestamp.replace('Z', '+00:00'))
    
    # Store prediction in database
    await crud.create_prediction(
        db=db,
        auid=request.auid,
        timestamp=timestamp,
        anomaly_detected=prediction_result["anomaly_detected"],
        failing_parts=prediction_result["failing_parts"],
    )
    
    # Retrieve historical predictions
    failure_service = get_failure_detection_service()
    historical_predictions = await crud.get_last_n_predictions(
        db=db,
        auid=request.auid,
        n=failure_service.window_size,
    )
    
    # Evaluate failure risk
    failure_evaluation = failure_service.evaluate_failure(historical_predictions)
    
    # Build response
    if debug:
        # Debug mode: include detailed predictions and history
        response = AnomalyDetectionDebugResponse(
            anomaly_detected=prediction_result["anomaly_detected"],
            failure_imminent=failure_evaluation["failure_imminent"],
            failing_parts=prediction_result["failing_parts"],
            auid=request.auid,
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
        # Normal mode: minimal response
        response = AnomalyDetectionResponse(
            anomaly_detected=prediction_result["anomaly_detected"],
            failure_imminent=failure_evaluation["failure_imminent"],
            failing_parts=prediction_result["failing_parts"],
            auid=request.auid,
        )
    
    return response
