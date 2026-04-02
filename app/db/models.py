"""SQLAlchemy database models."""

from datetime import datetime
from typing import List

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Prediction(Base):
    """Stores prediction history for anomaly detection per AUID."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    auid = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    anomaly_detected = Column(Boolean, nullable=False)
    failing_parts = Column(JSONB, nullable=False)  # Stores list of failing parts as JSON
    
    # Composite index for efficient queries on (auid, timestamp)
    __table_args__ = (
        Index('idx_auid_timestamp', 'auid', 'timestamp'),
    )

    def __repr__(self) -> str:
        return f"<Prediction(id={self.id}, auid={self.auid}, timestamp={self.timestamp}, anomaly={self.anomaly_detected})>"
