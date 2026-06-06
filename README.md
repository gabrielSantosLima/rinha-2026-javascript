# Rinha 2026 JavaScript

This repository contains a fraud detection solution for Rinha de Backend 2026. It is organized into two main projects:

- `api`: a Node.js/Express API that loads an ONNX model and exposes a fraud scoring endpoint.
- `research`: a Python project used to download reference datasets, train TensorFlow models, evaluate experiments, and export trained models.

## Repository structure

```text
.
├── api/
│   ├── src/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── model.onnx
│   └── README.md
├── research/
│   ├── setup.py
│   ├── training.py
│   ├── deploy.py
│   ├── environment.yml
│   └── README.md
└── README.md
```

## Requirements

Install the tools needed for the project you want to run.

### API requirements

- Node.js 24 or a compatible recent Node.js version
- pnpm 11 or a compatible pnpm version
- Docker and Docker Compose, if you want to run the load-balanced container setup

### Research requirements

- Conda or another compatible environment manager
- Python 3.11, as defined in `research/environment.yml`
- Network access to download Rinha reference resources during dataset setup

## Setup and run the API

From the repository root:

```bash
cd api
pnpm install
pnpm start
```

The API listens on `http://localhost:3333`.

Available endpoints:

- `GET /ready`: readiness endpoint.
- `POST /fraud-score`: receives transaction data, normalizes it, runs the ONNX model, and returns a fraud decision.

For development with automatic restart:

```bash
cd api
pnpm dev
```

For the Docker Compose setup with two API instances behind Nginx:

```bash
cd api
docker compose up --build
```

The load-balanced API is exposed on `http://localhost:9999`.

See `api/README.md` for more details.

## Setup and run the research project

From the repository root:

```bash
cd research
conda env create -f environment.yml
conda activate rinha
python setup.py
python training.py
```

The setup script downloads reference data into `research/data/`. The training script creates experiment directories with trained models and metrics.

To convert a trained Keras model to TFLite:

```bash
cd research
python deploy.py --model ./experiment_<timestamp>/model.keras
```

See `research/README.md` for more details.

## Model deployment flow

The typical workflow is:

1. Use `research/setup.py` to download and prepare the dataset.
2. Use `research/training.py` to train and evaluate a TensorFlow/Keras model.
3. Use `research/deploy.py` to export the trained model to TFLite.
4. Copy the TFLite model to `api/model.tflite`, if needed.
5. Run `api/convert_to_onnx.py` to generate `api/model.onnx`.
6. Start the API so it can load `api/model.onnx` and serve predictions.

## Notes

- The API expects `api/model.onnx` to exist before startup.
- Docker Compose uses the image name configured in `api/docker-compose.yml` and exposes Nginx on port `9999`.
- Generated data, experiment outputs, and model artifacts should be handled carefully because they can be large.
