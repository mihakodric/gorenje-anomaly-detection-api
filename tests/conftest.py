"""Pytest configuration and fixtures."""

import pytest_asyncio
from sqlalchemy import delete

from app.db.database import AsyncSessionLocal, close_db, init_db
from app.db.models import Prediction
from app.services.inference import close_inference_service, init_inference_service


@pytest_asyncio.fixture(scope="session", autouse=True)
async def initialize_app_resources():
    """Initialize shared app resources used by API tests."""
    await init_db()
    init_inference_service()
    yield
    close_inference_service()
    await close_db()


@pytest_asyncio.fixture(autouse=True)
async def reset_db():
    """Clear persisted predictions between tests."""
    async with AsyncSessionLocal() as session:
        await session.execute(delete(Prediction))
        await session.commit()
    yield
