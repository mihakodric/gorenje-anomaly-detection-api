# Gorenje Anomaly Detection API - Production Ready

This is a **production-ready FastAPI service** that exposes ML models for anomaly detection and failure prediction for washing machine cycles, with persistent prediction history.

## 🚀 Quick Start

### Using Docker

1. **Create `.env` file** from the example:
   ```bash
   cp .env.example .env
   ```

2. **Start the services**:
   ```bash
   docker-compose up --build
   ```

3. **Access the API**:
   - API: http://localhost:8000
   - Swagger Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## 📡 API Endpoints

### POST `/detect_anomaly`

Detect anomalies and predict failures for washing machine cycle data.

**Request:**
```json
{
  "auid": "string",
  "timestamp": "ISO8601 string",
  "data_102_0": { ... },
  "data_102_65": { ... }
}
```

**Response:**
```json
{
  "anomaly_detected": true,
  "failure_imminent": true,
  "failing_parts": ["heater"],
  "auid": "string"
}
```

**Debug Mode** (`?debug=true`):
```json
{
  "anomaly_detected": true,
  "failure_imminent": true,
  "failing_parts": ["heater"],
  "auid": "string",
  "predictions": {
    "heater": { ... },
    "pump": { ... },
    "motor": { ... }
  },
  "history": { ... }
}
```

### GET `/health`

Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

Use Swagger UI at `http://localhost:8000/docs` for interactive request examples.

## 🧠 How It Works

### 1. ML Inference
- Runs Keras models for **heater**, **pump**, and **motor** components
- Compares predictions vs actual values
- Flags anomalies based on residual thresholds

### 2. Persistent History
- Stores each prediction in PostgreSQL
- Tracks: `auid`, `timestamp`, `anomaly_detected`, `failing_parts`

### 3. Failure Detection (Temporal Aggregation)
Configurable strategies via `.env`:

**Threshold Strategy** (`REQUIRE_CONSECUTIVE=false`):
- Retrieve last `WINDOW_SIZE` predictions
- If ≥ `THRESHOLD_COUNT` are anomalies → `failure_imminent=true`

**Consecutive Strategy** (`REQUIRE_CONSECUTIVE=true`):
- If ≥ `CONSECUTIVE_THRESHOLD` consecutive anomalies → `failure_imminent=true`

## ⚙️ Configuration

All settings in `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection string |
| `WINDOW_SIZE` | `10` | Number of past cycles to consider |
| `THRESHOLD_COUNT` | `8` | Min anomalies for failure (threshold mode) |
| `CONSECUTIVE_THRESHOLD` | `3` | Min consecutive anomalies (consecutive mode) |
| `REQUIRE_CONSECUTIVE` | `false` | Use consecutive vs threshold strategy |
| `HEATER_LIMIT` | _(auto)_ | Heater anomaly threshold (optional) |
| `PUMP_LIMIT` | _(auto)_ | Pump anomaly threshold (optional) |
| `MOTOR_LIMIT` | _(auto)_ | Motor anomaly threshold (optional) |

## 🗂️ Project Structure

```
gorenje-anomaly-detection-api/
├── app/
│   ├── api/
│   │   └── routes.py           # API endpoints
│   ├── core/
│   │   └── config.py           # Configuration management
│   ├── db/
│   │   ├── models.py           # SQLAlchemy models
│   │   ├── database.py         # Database connection
│   │   └── crud.py             # Database operations
│   ├── schemas/
│   │   ├── request.py          # Request models
│   │   └── response.py         # Response models
│   ├── services/
│   │   ├── inference.py        # ML inference service
│   │   └── failure_logic.py    # Failure detection logic
│   └── main.py                 # FastAPI application
├── utils/
│   ├── heater/                 # Heater model & scalers
│   ├── pump/                   # Pump model & scalers
│   ├── motor/                  # Motor model & scalers
│   ├── model.py               # Original model logic
│   └── columns.py             # Feature definitions
├── tests/
│   ├── test_api.py            # API tests
│   └── test_failure_logic.py  # Logic tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
```

## 🧪 Testing

### Run Tests
```bash
# Start the database first
docker-compose up -d db

# Run the test suite in a one-off API container with the repo mounted
docker-compose run --rm -v "${PWD}:/app" api pytest tests/
```

Running `pytest tests/` directly inside the existing `api` container will not work as-is, because the image is built to run the application and does not copy the `tests/` directory into the container.

### Test Failure Detection
Send 10 consecutive requests with the same AUID:
```bash
for i in {1..10}; do
  curl -X POST http://localhost:8000/detect_anomaly \
    -H "Content-Type: application/json" \
    -d '{ ... }'
done
```

The 8th request should trigger `failure_imminent=true` (with default config).

## 🐳 Docker Commands

```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down

# Remove volumes (reset database)
docker-compose down -v
```

##  Monitoring

- **Health Check**: `GET /health`
- **Logs**: Check `docker-compose logs -f api`
- **Database**: Connect to PostgreSQL on `localhost:5432`

## 🔒 Production Considerations

1. **Environment Variables**: Never commit `.env` file
2. **Database**: Use managed PostgreSQL (Azure, AWS RDS) in production
3. **Secrets**: Use Docker secrets or environment variable injection
4. **SSL**: Configure reverse proxy (nginx) with SSL certificates
5. **Rate Limiting**: Add rate limiting middleware for production
6. **Monitoring**: Integrate with Prometheus/Grafana
7. **Logging**: Configure structured logging (JSON format)

## 📝 Migration from CLI

Original CLI tool (`main.py`) is preserved. The API wraps the same ML logic:
- `utils/model.py` - Original HealthCheck class
- `app/services/inference.py` - API-adapted inference service

## 📮 API Documentation

Interactive API docs available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
