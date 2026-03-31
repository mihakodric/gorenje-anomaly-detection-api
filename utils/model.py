import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
import json
from datetime import datetime
from typing import Tuple, Dict, Any, List

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import joblib
import keras
import tensorflow as tf
import numpy as np
import pytz
from sklearn.exceptions import InconsistentVersionWarning
from dotenv import load_dotenv

from utils.columns import region_0_cols, region_65_cols

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
tf.get_logger().setLevel("ERROR")

load_dotenv()


CYCLE_SETTINGS_KEY = "data_102_0"
CYCLE_RESULT_KEY = "data_102_65"

HEATER_TARGET_KEY = "diff_actuator13worktimeinseconds"
PUMP_TARGET_KEY = "diff_actuator1worktimeinseconds"
MOTOR_TARGET_KEY = "diff_totalmotorenergyconsumtion"

class Parser:
    """Parse cycle settings and result data from a raw JSON payload.

    The input file is expected to contain top-level keys
    :data:`CYCLE_SETTINGS_KEY` and :data:`CYCLE_RESULT_KEY` whose values are
    flat dictionaries of sensor values. Only the columns listed in
    :data:`region_0_cols` and :data:`region_65_cols` are kept.
    """

    def __call__(self, json_path: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """Parse a JSON file and return the processed content.

        This is a convenience wrapper around :meth:`_parse` so that the
        instance can be used like a function.

        Args:
            json_path: Path to the JSON file on disk.

        Returns:
            A tuple ``(auid, cycle_settings, cycle_result)`` where:

            * ``auid`` is the appliance or cycle identifier.
            * ``cycle_settings`` contains the filtered settings/features
              from :data:`CYCLE_SETTINGS_KEY`.
            * ``cycle_result`` contains the filtered result values from
              :data:`CYCLE_RESULT_KEY`.
        """

        return self._parse(json_path)

    def _parse(self, json_path: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """Low-level implementation for parsing the JSON input file.

        Args:
            json_path: Path to the JSON file to load.

        Returns:
            A tuple ``(auid, cycle_settings, cycle_result)`` as described
            in :meth:`__call__`.
        """

        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        auid = data.get("auid")
        cycle_settings_raw = data.get(CYCLE_SETTINGS_KEY, {})
        cycle_result_raw = data.get(CYCLE_RESULT_KEY, {})

        cycle_settings = {
            key: cycle_settings_raw[key]
            for key in region_0_cols
            if key in cycle_settings_raw
        }
        cycle_result = {
            key: cycle_result_raw[key]
            for key in region_65_cols
            if key in cycle_result_raw
        }

        cycle_settings["diff_loadweightedcycles"] = cycle_result.pop(
            "diff_loadweightedcycles",
            cycle_settings.get("diff_loadweightedcycles"),
        )
        cycle_settings["diff_cumulativeeccentricload"] = cycle_result.pop(
            "diff_cumulativeeccentricload",
            cycle_settings.get("diff_cumulativeeccentricload"),
        )

        return auid, cycle_settings, cycle_result

class HealthCheck:
    """Run model-based health checks for heater, pump, and motor.

    The class encapsulates loading of the Keras models, associated
    feature/target scalers, and residual statistics. Calling an instance
    performs a single-cycle evaluation and writes the result to a JSON
    file inside the ``results/`` directory.
    """

    def __init__(self) -> None:
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

        self._compile_model(self.heater_model)
        self._compile_model(self.pump_model)
        self._compile_model(self.motor_model)

        self.scaler_X_heater = joblib.load("utils/heater/scaler_x.save")
        self.scaler_X_pump = joblib.load("utils/pump/scaler_x.save")
        self.scaler_X_motor = joblib.load("utils/motor/scaler_x.save")

        self.scaler_Y_heater = joblib.load("utils/heater/scaler_y.save")
        self.scaler_Y_pump = joblib.load("utils/pump/scaler_y.save")
        self.scaler_Y_motor = joblib.load("utils/motor/scaler_y.save")

        self.std_heater = joblib.load("utils/heater/heater_residual_std.save")
        self.std2_heater = joblib.load("utils/heater/heater_residual_std_2.save")
        self.mean_heater = joblib.load("utils/heater/heater_residual_mean.save")

        self.std_pump = joblib.load("utils/pump/pump_residual_std.save")
        self.std2_pump = joblib.load("utils/pump/pump_residual_std_2.save")
        self.mean_pump = joblib.load("utils/pump/pump_residual_mean.save")

        self.std_motor = joblib.load("utils/motor/motor_residual_std.save")
        self.std2_motor = joblib.load("utils/motor/motor_residual_std_2.save")
        self.mean_motor = joblib.load("utils/motor/motor_residual_mean.save")

        heater_limit_env = os.getenv("HEATER_LIMIT")
        pump_limit_env = os.getenv("PUMP_LIMIT")
        motor_limit_env = os.getenv("MOTOR_LIMIT")

        def _get_limit(env_value, mean, std):
            if env_value is not None:
                stripped = env_value.strip()
                if stripped:
                    try:
                        return float(stripped)
                    except ValueError:
                        pass

            return float(mean + 2 * std)

        self.heater_limit = _get_limit(heater_limit_env, self.mean_heater, self.std_heater)
        self.pump_limit = _get_limit(pump_limit_env, self.mean_pump, self.std_pump)
        self.motor_limit = _get_limit(motor_limit_env, self.mean_motor, self.std_motor)

    @staticmethod
    def _compile_model(model: keras.Model) -> None:
        """Compile a Keras model with the common configuration.

        The configuration mirrors the setup used during training and is
        sufficient for running forward passes and computing metrics.

        Args:
            model: Instantiated Keras model to be compiled in-place.
        """

        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.Huber(),
            metrics=[
                keras.metrics.MeanSquaredError(name="mean_squared_error"),
                keras.metrics.MeanAbsoluteError(name="mean_absolute_error"),
                keras.metrics.RootMeanSquaredError(name="root_mean_squared_error"),
            ],
        )

    def __call__(
        self,
        cycle_settings: Dict[str, Any],
        cycle_result: Dict[str, Any],
        auid: str | None = None,
    ) -> None:
        """Run the health check for a single cycle.

        This is a thin wrapper around :meth:`_prediction` for a more
        convenient, function-like usage of the class.

        Args:
            cycle_settings: Mapping of feature names to values used as
                model input. Typically produced by :class:`Parser`.
            cycle_result: Mapping of observed target values (heater,
                pump, motor) for the same cycle.
            auid: Optional identifier of the appliance or cycle, used
                only for naming the output file.
        """

        self._prediction(cycle_settings, cycle_result, auid)

    def _prediction(
        self,
        x_dict: Dict[str, Any],
        y_true: Dict[str, Any],
        auid: str | None = None,
    ) -> None:
        """Generate predictions and residuals for the three components.

        Args:
            x_dict: Pre-processed feature values (same ordering as
                ``region_0_cols``) used as model input.
            y_true: Observed target values containing at least the
                keys :data:`HEATER_TARGET_KEY`, :data:`PUMP_TARGET_KEY`,
                and :data:`MOTOR_TARGET_KEY`.
            auid: Optional identifier propagated to the output JSON.
        """

        features = np.array(list(x_dict.values()), dtype=float).reshape(1, -1)

        x_heater = self.scaler_X_heater.transform(features)
        x_pump = self.scaler_X_pump.transform(features)
        x_motor = self.scaler_X_motor.transform(features)

        y_pred_scaled_heater = self.heater_model.predict(x_heater).reshape(-1, 1)
        y_pred_scaled_pump = self.pump_model.predict(x_pump).reshape(-1, 1)
        y_pred_scaled_motor = self.motor_model.predict(x_motor).reshape(-1, 1)

        y_pred_heater = float(
            self.scaler_Y_heater.inverse_transform(y_pred_scaled_heater).flatten()[0]
        )
        y_pred_pump = float(
            self.scaler_Y_pump.inverse_transform(y_pred_scaled_pump).flatten()[0]
        )
        y_pred_motor = float(
            self.scaler_Y_motor.inverse_transform(y_pred_scaled_motor).flatten()[0]
        )

        residual_heater = abs(y_pred_heater - float(y_true[HEATER_TARGET_KEY]))
        residual_pump = abs(y_pred_pump - float(y_true[PUMP_TARGET_KEY]))
        residual_motor = abs(y_pred_motor - float(y_true[MOTOR_TARGET_KEY]))

        y_preds: List[float] = [y_pred_heater, y_pred_pump, y_pred_motor]
        residuals: List[float] = [residual_heater, residual_pump, residual_motor]
        means: List[float] = [self.mean_heater, self.mean_pump, self.mean_motor]
        stds: List[float] = [self.std_heater, self.std_pump, self.std_motor]
        std2s: List[float] = [self.std2_heater, self.std2_pump, self.std2_motor]

        self._generate_json(auid, y_preds, y_true, residuals, stds, std2s, means)

    def _generate_json(
        self,
        auid: str | None,
        y_preds: List[float],
        y_true: Dict[str, Any],
        residuals: List[float],
        stds: List[float],
        std2s: List[float],
        means: List[float],
    ) -> None:
        """Persist prediction results and metadata as a JSON report.

        The report contains model predictions, ground-truth targets,
        residuals, residual distribution statistics, and a simple
        anomaly flag with the list of failing components. The file is
        written to the ``results/`` directory and named using the
        provided ``auid`` (if any) and a UTC timestamp.

        Args:
            auid: Optional identifier that is embedded into the output
                structure and used in the filename prefix.
            y_preds: Predicted values for heater, pump, and motor in
                that order.
            y_true: Mapping of true target values.
            residuals: Absolute residuals between prediction and truth
                for each component.
            stds: Per-component residual standard deviations (1σ).
            std2s: Per-component residual thresholds corresponding to
                2σ.
            means: Per-component residual means.
        """

        limits = [self.heater_limit, self.pump_limit, self.motor_limit]

        anomaly = any(
            residual > limit
            for residual, limit in zip(residuals, limits)
        )

        failing_parts: List[str] = []
        if residuals[0] > limits[0]:
            failing_parts.append("heater")
        if residuals[1] > limits[1]:
            failing_parts.append("pump")
        if residuals[2] > limits[2]:
            failing_parts.append("motor")

        result = {
            "auid": auid,
            "predictions": {
                "heater": {
                    "predicted_value": y_preds[0],
                    "true_value": y_true[HEATER_TARGET_KEY],
                    "residual": residuals[0],
                    "σ": stds[0],
                    "2σ": std2s[0],
                    "mean": means[0],
                    "defined_limit": self.heater_limit,
                },
                "pump": {
                    "predicted_value": y_preds[1],
                    "true_value": y_true[PUMP_TARGET_KEY],
                    "residual": residuals[1],
                    "σ": stds[1],
                    "2σ": std2s[1],
                    "mean": means[1],
                    "defined_limit": self.pump_limit,
                },
                "motor": {
                    "predicted_value": y_preds[2],
                    "true_value": y_true[MOTOR_TARGET_KEY],
                    "residual": residuals[2],
                    "σ": stds[2],
                    "2σ": std2s[2],
                    "mean": means[2],
                    "defined_limit": self.motor_limit,
                },
            },
            "anomaly_detected": anomaly,
            "failing_parts": failing_parts,
        }

        timestamp = datetime.now(pytz.utc).strftime("%Y-%m-%d_%H-%M-%S")

        filename = f"results/{auid}_{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=4, ensure_ascii=False)
