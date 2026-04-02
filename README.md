# Gorenje Anomaly Detection API

Production-ready FastAPI service for washing machine anomaly detection and failure prediction with persistent history.

## Quick Start

```bash
# 1. Setup environment
cp .env.example .env

# 2. Start services
docker-compose up --build -d

# 3. Test API
curl http://localhost:8000/health
```

**API Documentation**: http://localhost:8000/docs

## Usage

**Detect Anomaly:**
```bash
curl -X POST http://localhost:8000/detect_anomaly \
  -H "Content-Type: application/json" \
  -d @input.json
```

**Response:**
```json
{
  "anomaly_detected": true,
  "failure_imminent": false,
  "failing_parts": ["pump"],
  "auid": "0000000000007442320000202400044930020"
}
```

**Debug Mode** (detailed predictions + history):
```bash
curl http://localhost:8000/detect_anomaly?debug=true -d @input.json
```

## How It Works

1. **ML Models** predict expected values for heater, pump, motor components
2. **Anomaly Detection** flags components exceeding residual thresholds
3. **Persistent Storage** saves each prediction to PostgreSQL
4. **Failure Prediction** analyzes recent history (e.g., 8/10 anomalies → failure imminent)

## Configuration

Edit [.env](.env) to customize:

```bash
WINDOW_SIZE=10              # History window for failure detection
THRESHOLD_COUNT=8           # Min anomalies for failure warning
HEATER_LIMIT=150           # Custom thresholds (optional)
PUMP_LIMIT=100
MOTOR_LIMIT=2
```

## Input Format

See [input.json](input.json) for example. Required fields:
- `auid` - Appliance unique ID
- `timestamp` - ISO8601 timestamp
- `data_102_0` - Cycle settings (46 features)
- `data_102_65` - Cycle results (5 metrics: heater, pump, motor work times)

## 🛠️ Development

**Local testing** (without Docker):
```bash
pip install -r requirements.txt
python main.py --filename input.json  # CLI mode
```

**Stop services:**
```bash
docker-compose down
```

**View logs:**
```bash
docker logs anomaly-detection-api
```

See [API_README.md](API_README.md) for detailed documentation.
