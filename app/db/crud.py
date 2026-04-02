"""CRUD operations for database models."""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Prediction


async def create_prediction(
    db: AsyncSession,
    auid: str,
    timestamp: datetime,
    anomaly_detected: bool,
    failing_parts: List[str],
) -> Prediction:
    """Create a new prediction record.
    
    Args:
        db: Database session.
        auid: Appliance unique identifier.
        timestamp: Timestamp of the prediction.
        anomaly_detected: Whether an anomaly was detected.
        failing_parts: List of failing component names.
        
    Returns:
        The created Prediction object.
    """
    prediction = Prediction(
        auid=auid,
        timestamp=timestamp,
        anomaly_detected=anomaly_detected,
        failing_parts=failing_parts,
    )
    db.add(prediction)
    await db.commit()
    await db.refresh(prediction)
    return prediction


async def get_last_n_predictions(
    db: AsyncSession,
    auid: str,
    n: int,
) -> List[Prediction]:
    """Retrieve the last N predictions for a given AUID, ordered by timestamp descending.
    
    Args:
        db: Database session.
        auid: Appliance unique identifier.
        n: Number of recent predictions to retrieve.
        
    Returns:
        List of Prediction objects, ordered by timestamp (most recent first).
    """
    stmt = (
        select(Prediction)
        .where(Prediction.auid == auid)
        .order_by(Prediction.timestamp.desc())
        .limit(n)
    )
    result = await db.execute(stmt)
    predictions = result.scalars().all()
    return list(predictions)


async def get_prediction_by_id(
    db: AsyncSession,
    prediction_id: int,
) -> Optional[Prediction]:
    """Get a specific prediction by ID.
    
    Args:
        db: Database session.
        prediction_id: Primary key of the prediction.
        
    Returns:
        Prediction object if found, None otherwise.
    """
    stmt = select(Prediction).where(Prediction.id == prediction_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def count_predictions_by_auid(
    db: AsyncSession,
    auid: str,
) -> int:
    """Count total predictions for a given AUID.
    
    Args:
        db: Database session.
        auid: Appliance unique identifier.
        
    Returns:
        Total count of predictions.
    """
    stmt = select(Prediction).where(Prediction.auid == auid)
    result = await db.execute(stmt)
    return len(result.scalars().all())
