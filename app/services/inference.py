"""ML inference service for anomaly detection.

Wraps the existing HealthCheck model logic and adapts it for API usage.
"""

import os
from typing import Dict, Any, List, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
import numpy as np
import joblib
import keras
import tensorflow as tf
from sklearn.exceptions import InconsistentVersionWarning

from utils.columns import region_0_cols, region_65_cols
from app.core.config import settings

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
tf.get_logger().setLevel("ERROR")

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


CYCLE_SETTINGS_KEY = "data_102_0"
CYCLE_RESULT_KEY = "data_102_65"

HEATER_TARGET_KEY = "diff_actuator13worktimeinseconds"
PUMP_TARGET_KEY = "diff_actuator1worktimeinseconds"
MOTOR_TARGET_KEY = "diff_totalmotorenergyconsumtion"


class InferenceService:
    """ML inference service for washing machine anomaly detection.
    
    Loads models at initialization and provides prediction methods
    that return structured dictionaries instead of writing to files.
    """
    
    def __init__(self) -> None:
        """Initialize models, scalers, and residual statistics."""
        # Load models
        self.heater_model = keras.models.load_model(
            "utils/heater/heater.h5",
            compile=False,
        )
        self.pump_model = keras.models.load_model(
            "utils/pump/pump.h5",
            compile=False,
        )
        self.motor_model = keras.models.load_model(
            "utils/motor/motor.h5",
            compile=False,
        )

        # Compile models
        self._compile_model(self.heater_model)
        self._compile_model(self.pump_model)
        self._compile_model(self.motor_model)

        # Load scalers
        self.scaler_X_heater = joblib.load("utils/heater/scaler_x.save")
        self.scaler_X_pump = joblib.load("utils/pump/scaler_x.save")
        self.scaler_X_motor = joblib.load("utils/motor/scaler_x.save")

        self.scaler_Y_heater = joblib.load("utils/heater/scaler_y.save")
        self.scaler_Y_pump = joblib.load("utils/pump/scaler_y.save")
        self.scaler_Y_motor = joblib.load("utils/motor/scaler_y.save")

        # Load residual statistics
        self.std_heater = joblib.load("utils/heater/heater_residual_std.save")
        self.std2_heater = joblib.load("utils/heater/heater_residual_std_2.save")
        self.mean_heater = joblib.load("utils/heater/heater_residual_mean.save")

        self.std_pump = joblib.load("utils/pump/pump_residual_std.save")
        self.std2_pump = joblib.load("utils/pump/pump_residual_std_2.save")
        self.mean_pump = joblib.load("utils/pump/pump_residual_mean.save")

        self.std_motor = joblib.load("utils/motor/motor_residual_std.save")
        self.std2_motor = joblib.load("utils/motor/motor_residual_std_2.save")
        self.mean_motor = joblib.load("utils/motor/motor_residual_mean.save")

        # Calculate limits from settings or use mean + 2*std
        self.heater_limit = self._get_limit(
            settings.heater_limit, self.mean_heater, self.std_heater
        )
        self.pump_limit = self._get_limit(
            settings.pump_limit, self.mean_pump, self.std_pump
        )
        self.motor_limit = self._get_limit(
            settings.motor_limit, self.mean_motor, self.std_motor
        )

    @staticmethod
    def _compile_model(model: keras.Model) -> None:
        """Compile a Keras model with standard configuration."""
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.Huber(),
            metrics=[
                keras.metrics.MeanSquaredError(name="mean_squared_error"),
                keras.metrics.MeanAbsoluteError(name="mean_absolute_error"),
                keras.metrics.RootMeanSquaredError(name="root_mean_squared_error"),
            ],
        )

    @staticmethod
    def _get_limit(env_value: float | None, mean: float, std: float) -> float:
        """Calculate anomaly threshold from config or default to mean + 2*std."""
        if env_value is not None:
            return float(env_value)
        return float(mean + 2 * std)

    def parse_input(
        self, 
        cycle_settings_raw: Dict[str, Any], 
        cycle_result_raw: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Parse and filter input data based on expected columns.
        
        Args:
            cycle_settings_raw: Raw data_102_0 dictionary.
            cycle_result_raw: Raw data_102_65 dictionary.
            
        Returns:
            Tuple of (filtered_settings, filtered_results).
        """
        # Filter settings based on region_0_cols
        cycle_settings = {
            key: cycle_settings_raw[key]
            for key in region_0_cols
            if key in cycle_settings_raw
        }
        
        # Filter results based on region_65_cols
        cycle_result = {
            key: cycle_result_raw[key]
            for key in region_65_cols
            if key in cycle_result_raw
        }

        # Move shared fields from result to settings (matching original logic)
        cycle_settings["diff_loadweightedcycles"] = cycle_result.pop(
            "diff_loadweightedcycles",
            cycle_settings.get("diff_loadweightedcycles"),
        )
        cycle_settings["diff_cumulativeeccentricload"] = cycle_result.pop(
            "diff_cumulativeeccentricload",
            cycle_settings.get("diff_cumulativeeccentricload"),
        )

        return cycle_settings, cycle_result

    def predict(
        self,
        cycle_settings: Dict[str, Any],
        cycle_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run inference and return prediction results.
        
        Args:
            cycle_settings: Filtered cycle settings (features).
            cycle_result: Filtered cycle results (targets).
            
        Returns:
            Dictionary containing:
                - anomaly_detected: bool
                - failing_parts: List[str]
                - predictions: Dict with per-component details
        """
        # Convert features to numpy array
        features = np.array(list(cycle_settings.values()), dtype=float).reshape(1, -1)

        # Scale features for each model
        x_heater = self.scaler_X_heater.transform(features)
        x_pump = self.scaler_X_pump.transform(features)
        x_motor = self.scaler_X_motor.transform(features)

        # Get predictions
        y_pred_scaled_heater = self.heater_model.predict(x_heater, verbose=0).reshape(-1, 1)
        y_pred_scaled_pump = self.pump_model.predict(x_pump, verbose=0).reshape(-1, 1)
        y_pred_scaled_motor = self.motor_model.predict(x_motor, verbose=0).reshape(-1, 1)

        # Inverse transform predictions
        y_pred_heater = float(
            self.scaler_Y_heater.inverse_transform(y_pred_scaled_heater).flatten()[0]
        )
        y_pred_pump = float(
            self.scaler_Y_pump.inverse_transform(y_pred_scaled_pump).flatten()[0]
        )
        y_pred_motor = float(
            self.scaler_Y_motor.inverse_transform(y_pred_scaled_motor).flatten()[0]
        )

        # Get true values
        y_true_heater = float(cycle_result[HEATER_TARGET_KEY])
        y_true_pump = float(cycle_result[PUMP_TARGET_KEY])
        y_true_motor = float(cycle_result[MOTOR_TARGET_KEY])

        # Calculate residuals
        residual_heater = abs(y_pred_heater - y_true_heater)
        residual_pump = abs(y_pred_pump - y_true_pump)
        residual_motor = abs(y_pred_motor - y_true_motor)

        # Determine anomalies
        heater_anomaly = residual_heater > self.heater_limit
        pump_anomaly = residual_pump > self.pump_limit
        motor_anomaly = residual_motor > self.motor_limit

        anomaly_detected = heater_anomaly or pump_anomaly or motor_anomaly

        failing_parts: List[str] = []
        if heater_anomaly:
            failing_parts.append("heater")
        if pump_anomaly:
            failing_parts.append("pump")
        if motor_anomaly:
            failing_parts.append("motor")

        # Build detailed response
        result = {
            "anomaly_detected": anomaly_detected,
            "failing_parts": failing_parts,
            "predictions": {
                "heater": {
                    "predicted_value": y_pred_heater,
                    "true_value": y_true_heater,
                    "residual": residual_heater,
                    "sigma": float(self.std_heater),
                    "two_sigma": float(self.std2_heater),
                    "mean": float(self.mean_heater),
                    "defined_limit": self.heater_limit,
                    "is_anomaly": heater_anomaly,
                },
                "pump": {
                    "predicted_value": y_pred_pump,
                    "true_value": y_true_pump,
                    "residual": residual_pump,
                    "sigma": float(self.std_pump),
                    "two_sigma": float(self.std2_pump),
                    "mean": float(self.mean_pump),
                    "defined_limit": self.pump_limit,
                    "is_anomaly": pump_anomaly,
                },
                "motor": {
                    "predicted_value": y_pred_motor,
                    "true_value": y_true_motor,
                    "residual": residual_motor,
                    "sigma": float(self.std_motor),
                    "two_sigma": float(self.std2_motor),
                    "mean": float(self.mean_motor),
                    "defined_limit": self.motor_limit,
                    "is_anomaly": motor_anomaly,
                },
            },
        }

        return result


# Singleton instance (will be initialized at app startup)
_inference_service: InferenceService | None = None


def get_inference_service() -> InferenceService:
    """Get the singleton inference service instance.
    
    Returns:
        InferenceService instance.
        
    Raises:
        RuntimeError: If service hasn't been initialized.
    """
    global _inference_service
    if _inference_service is None:
        raise RuntimeError("InferenceService not initialized. Call init_inference_service() first.")
    return _inference_service


def init_inference_service() -> None:
    """Initialize the singleton inference service."""
    global _inference_service
    _inference_service = InferenceService()


def close_inference_service() -> None:
    """Clean up inference service resources."""
    global _inference_service
    _inference_service = None
