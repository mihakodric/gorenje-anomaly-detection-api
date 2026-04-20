"""Application configuration management.

Loads all configuration from environment variables with sensible defaults.
"""

import json
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database settings
    database_url: str = "postgresql+asyncpg://postgres:postgres@db:5432/anomaly_detection"
    
    # Failure detection thresholds
    window_size: int = 10
    threshold_count: int = 8
    consecutive_threshold: int = 3
    require_consecutive: bool = False
    
    # Model residual limits (optional, will use mean + 2*std if not provided)
    heater_limit: Optional[float] = None
    pump_limit: Optional[float] = None
    motor_limit: Optional[float] = None

    # Component display colors
    component_ok_color: str = "#669900"
    component_warning_color: str = "#FFCC00"
    component_failing_color: str = "#CC0000"
    
    # API settings
    api_title: str = "Washing Machine Anomaly Detection API"
    api_version: str = "1.0.0"
    api_description: str = "Production-ready FastAPI service for ML-based anomaly detection and failure prediction"
    
    # CORS settings (optional)
    cors_origins: str = "*"
    
    # Logging
    log_level: str = "INFO"

    @property
    def cors_origins_list(self) -> list[str]:
        """Accept *, comma-separated strings, or a JSON list in the env file."""
        raw = self.cors_origins.strip()
        if not raw or raw == "*":
            return ["*"]

        if raw.startswith("[") and raw.endswith("]"):
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(origin).strip() for origin in parsed if str(origin).strip()]

        return [origin.strip() for origin in raw.split(",") if origin.strip()]


# Singleton instance
settings = Settings()
