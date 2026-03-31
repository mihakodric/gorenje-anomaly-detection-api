# Flowmaster – Cycle Health Check

This project provides a small, self-contained pipeline for running model-based health checks on washing machine cycles. Given a JSON file containing cycle settings and cycle results, it predicts expected behavior for key components (heater, pump, motor), computes residuals, and flags potential anomalies.

The main entry point is [main.py](main.py), which loads trained models and scalers from the `utils/` directory and writes a structured JSON report into the `results/` folder.

---

## Project Structure

- [main.py](main.py) – CLI entry point that parses arguments, reads the input JSON, and triggers the health check.
- [input.json](input.json) – Example input file containing cycle settings and results in the expected format.
- [utils/model.py](utils/model.py) – Core logic:
  - `Parser` – extracts and preprocesses cycle settings/results from the input JSON.
  - `HealthCheck` – loads models and scalers, performs predictions, computes residuals, and generates output reports.
- [utils/columns.py](utils/columns.py) – Column lists used to select the relevant features from the raw JSON.
- [utils/requirements.txt](utils/requirements.txt) – Python dependencies for running the models.
- [utils/heater](utils/heater), [utils/pump](utils/pump), [utils/motor](utils/motor) – Trained Keras models, scalers, and residual statistics for each component.
- [results/](results/) – Output directory where prediction results are stored as timestamped JSON files.
- [tutorial.ipynb](tutorial.ipynb) – Jupyter notebook for interactive experimentation and exploration.

---

## Installation

1. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```

2. **Install the dependencies** listed in `utils/requirements.txt`:

   ```bash
   pip install -r utils/requirements.txt
   ```

> Note: The project relies on TensorFlow/Keras and scikit-learn. Depending on your hardware and existing environment, you may want to install GPU-enabled TensorFlow or pin specific versions.

---

## Input Format

The main script expects a JSON file with the following structure (simplified):

```json
{
  "auid": "0000000000007442320000202400044930020",
  "data_102_0": {
    "status": 1,
    "error_3": 0,
    "error_6": 0,
    "selected_program_id_status": 5,
    "selected_program_duration_in_minutes": 90,
    "diff_loadweightedcycles": 123,
    "diff_cumulativeeccentricload": 456,
    "...": "other settings used as model features"
  },
  "data_102_65": {
    "diff_loadweightedcycles": 123,
    "diff_cumulativeeccentricload": 456,
    "diff_actuator1worktimeinseconds": 10.2,
    "diff_actuator13worktimeinseconds": 32.5,
    "diff_totalmotorenergyconsumtion": 150.7
  }
}
```

Relevant feature names for `data_102_0` and `data_102_65` are defined in [utils/columns.py](utils/columns.py) as `region_0_cols` and `region_65_cols`.

- `auid` – Unique identifier of the appliance or cycle.
- `data_102_0` – Cycle settings and configuration values.
- `data_102_65` – Measured cycle result values for the same cycle.

---

## Usage

After installing dependencies and preparing an input file, run:

```bash
python main.py --filename path/to/your_input.json
```

`main.py` will:

1. Parse the input JSON and extract the relevant fields.
2. Run the heater, pump, and motor models on the preprocessed features.
3. Compute residuals (absolute error between predicted and true values).
4. Determine whether an anomaly is present based on learned residual statistics.
5. Write a JSON report into the [results/](results/) directory.

The output filename has the form:

```text
results/<AUID>_<UTC_TIMESTAMP>.json
```

For example:

```text
results/0000000000007442320000202400044930020_2026-03-31_08-09-45.json
```

---

## Output JSON Structure

A typical output file looks like this (structure-only example):

```json
{
  "auid": "0000000000007442320000202400044930020",
  "predictions": {
    "heater": {
      "predicted_value": 30.1,
      "true_value": 32.5,
      "residual": 2.4,
      "σ": 1.2,
      "2σ": 2.4,
      "mean": 0.5
    },
    "pump": {
      "predicted_value": 9.8,
      "true_value": 10.2,
      "residual": 0.4,
      "σ": 0.7,
      "2σ": 1.4,
      "mean": 0.3
    },
    "motor": {
      "predicted_value": 148.2,
      "true_value": 150.7,
      "residual": 2.5,
      "σ": 1.5,
      "2σ": 3.0,
      "mean": 0.8
    }
  },
  "anomaly_detected": true,
  "failing_parts": [
    "heater",
    "motor"
  ]
}
```

- `predicted_value` – Model prediction for the target quantity.
- `true_value` – Actual measured value from the cycle results.
- `residual` – Absolute difference between predicted and true values.
- `σ`, `2σ`, `mean` – Residual distribution parameters learned from historical data.
- `anomaly_detected` – Boolean flag indicating whether any component residual exceeds the configured threshold.
- `failing_parts` – List of component names (`"heater"`, `"pump"`, `"motor"`) whose residual exceeds the 2σ threshold.

---

## Notebook

To explore the models and pipeline interactively, open [tutorial.ipynb](tutorial.ipynb) in Jupyter or VS Code and run the cells. Ensure the same virtual environment (with the installed dependencies) is selected as the notebook kernel.

---

## Notes

- The models and statistics in `utils/heater`, `utils/pump`, and `utils/motor` are treated as opaque artifacts. Retraining them is beyond the scope of this repository.
- If you plan to integrate this into a larger system, consider wrapping `Parser` and `HealthCheck` from [utils/model.py](utils/model.py) in your own service or API layer.
