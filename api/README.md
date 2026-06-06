# API

This project is a Node.js/Express API for fraud detection. It loads `model.onnx` with `onnxruntime-node`, normalizes incoming transaction data, and returns a fraud score response.

## Requirements

- Node.js 24 or a compatible recent Node.js version
- pnpm 11 or a compatible pnpm version
- Docker and Docker Compose, optional for the containerized load-balanced setup

## Install dependencies

From this directory:

```bash
pnpm install
```

## Run locally

Start the API:

```bash
pnpm start
```

The API listens on:

```text
http://localhost:3333
```

For development with automatic restart:

```bash
pnpm dev
```

## Run with Docker Compose

The Docker Compose file starts two API containers and an Nginx load balancer.

```bash
docker compose up --build
```

The load-balanced API is exposed on:

```text
http://localhost:9999
```

To stop the containers:

```bash
docker compose down
```

## Endpoints

### `GET /ready`

Readiness endpoint.

Example:

```bash
curl -i http://localhost:3333/ready
```

When using Docker Compose:

```bash
curl -i http://localhost:9999/ready
```

### `POST /fraud-score`

Receives transaction data, normalizes the payload, runs the ONNX model, and returns a response with the approval decision and fraud score.

Example:

```bash
curl -X POST http://localhost:3333/fraud-score \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 100,
    "installments": 1,
    "requested_at": "2026-01-01T12:00:00Z"
  }'
```

The exact request payload should match the transaction shape expected by `src/use_cases/normalize_use_case.js`.

Response format:

```json
{
  "approved": true,
  "fraud_score": 0.95
}
```

## Model file

The API requires this file before startup:

```text
model.onnx
```

The model is loaded from the API project root by `src/use_cases/detect_fraud_use_case.js`.

## Convert a TFLite model to ONNX

If you have a `model.tflite` file in this directory, install the required Python converter dependencies in your environment and run:

```bash
python convert_to_onnx.py
```

This generates:

```text
model.onnx
```

## Important files

- `src/index.js`: Express application and route definitions.
- `src/use_cases/detect_fraud_use_case.js`: ONNX model loading and inference.
- `src/use_cases/normalize_use_case.js`: transaction normalization before inference.
- `Dockerfile`: container image definition.
- `docker-compose.yml`: two API instances plus Nginx load balancer.
- `nginx.conf`: Nginx upstream and proxy configuration.
