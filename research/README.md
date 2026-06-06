# Research

This project contains the research and model training workflow for the fraud detection solution. It downloads reference data, prepares a dataset, trains TensorFlow/Keras models, saves experiment metrics, and exports trained models for deployment.

## Requirements

- Conda or another compatible environment manager
- Python 3.11, as defined in `environment.yml`
- Network access to download the reference resources

## Create the environment

From this directory:

```bash
conda env create -f environment.yml
conda activate rinha
```

If the environment already exists and you need to update it:

```bash
conda env update -f environment.yml --prune
conda activate rinha
```

## Download and prepare data

Run:

```bash
python setup.py
```

The setup script downloads the following resources into `data/`:

- `references.json.gz`
- `mcc_risk.json`
- `normalization.json`

It also creates:

```text
data/references.csv
```

## Train a model

Run the default training workflow:

```bash
python training.py
```

The training script:

- Loads `data/references.csv`.
- Splits the dataset into train and test sets.
- Runs hyperparameter tuning with Keras Tuner.
- Trains a TensorFlow/Keras model.
- Evaluates the model.
- Saves metrics and plots.
- Saves the trained model as `model.keras` inside an experiment directory.

The output directory follows this pattern:

```text
experiment_<timestamp>/
├── model.keras
└── metrics/
    ├── metrics.txt
    ├── training_history.png
    ├── confusion_matrix.png
    ├── class_distribution.png
    ├── class_distribution_train_test.png
    └── correlation_matrix.png
```

## Training options

The training script supports command-line options:

```bash
python training.py \
  --batch-size 512 \
  --epochs 100 \
  --validation-split 0.3 \
  --test-split 0.4 \
  --patience 10 \
  --tuning_epochs 20 \
  --seed 12
```

Useful flags:

- `--only-plot-info`: generate dataset plots without running model training.
- `--remove-correlated-features`: remove selected correlated features before training.
- `--experiment-path <path>`: reuse or continue from a specific experiment directory.

Example with a custom experiment path:

```bash
python training.py --experiment-path ./experiment_manual
```

## Export a trained model to TFLite

After training, convert a Keras model to TFLite:

```bash
python deploy.py --model ./experiment_<timestamp>/model.keras
```

The output file is written to the current directory using this naming pattern:

```text
model.keras.tflite
```

## Use the model in the API

The API uses an ONNX model. A typical deployment flow is:

1. Train a model with `python training.py`.
2. Convert it to TFLite with `python deploy.py --model ./experiment_<timestamp>/model.keras`.
3. Copy or rename the TFLite file to `../api/model.tflite`.
4. From `../api`, run `python convert_to_onnx.py` to generate `model.onnx`.
5. Start the API.

## Important files

- `setup.py`: downloads and prepares reference data.
- `training.py`: trains and evaluates TensorFlow/Keras models.
- `deploy.py`: converts a Keras model to TFLite.
- `custom_metrics.py`: custom metrics used during training and model loading.
- `environment.yml`: Conda environment definition.
