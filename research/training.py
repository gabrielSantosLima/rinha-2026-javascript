import argparse
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import keras_tuner
import seaborn as sns
import prettytable
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
)
from custom_metrics import BinaryFBetaScore

from datetime import datetime
from tensorflow import keras
import matplotlib.pyplot as plt
import logging

import os

os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={os.environ.get('CONDA_PREFIX')}"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Num GPUs Available: %d", len(tf.config.list_physical_devices("GPU")))

# Experiment setup
CURRENT_PATH = os.getcwd()
DATASET_PATH = os.path.join(CURRENT_PATH, "data", "references.csv")

# Training setup
BATCH_SIZE = 512
EPOCHS = 100
VALIDATION_SPLIT = 0.3
TEST_SPLIT = 0.4
PATIENCE = 10
DEFAULT_TUNING_EPOCHS = 20
SEED = 12


def create_experiment_directory(default_experiment_path: str = None):
    if default_experiment_path:
        logging.info(f"Using experiment path: {default_experiment_path}")
        experiment_path = default_experiment_path
        experiment_time = os.path.basename(experiment_path).split("_")[-1]
    else:
        experiment_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_path = os.path.join(CURRENT_PATH, f"experiment_{experiment_time}")
    model_checkpoint = os.path.join(experiment_path, "model.keras")
    metrics_path = os.path.join(experiment_path, "metrics")
    if not os.path.exists(experiment_path):
        logging.info(f"Creating experiment directory at {experiment_path}")
        os.makedirs(experiment_path)
    if not os.path.exists(metrics_path):
        logging.info(f"Creating metrics directory at {metrics_path}")
        os.makedirs(metrics_path)
    logging.info(f"Model checkpoint will be saved to {model_checkpoint}")
    return experiment_path, model_checkpoint, metrics_path


def load_dataset():
    data = pd.read_csv(DATASET_PATH)
    logging.info(f"Loaded dataset with {len(data)} samples.")
    x = np.array(data["vector"].apply(eval).tolist())
    y = np.array(data["label"].apply(lambda x: 1 if x == "fraud" else 0).tolist())
    return x, y


def build_model(hp: keras_tuner.HyperParameters, input_shape: tuple):
    logging.info(
        f"Building model with input shape {input_shape} and hyperparameters {hp.values}"
    )
    model = keras.Sequential(
        [
            keras.layers.Dense(
                hp.Int("units1", min_value=64, max_value=256, step=64),
                activation="relu",
                input_shape=input_shape,
            ),
            keras.layers.Dense(
                hp.Int("units2", min_value=32, max_value=128, step=32),
                activation="relu",
            ),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.AUC(name="auc"),
            BinaryFBetaScore(beta=1.0, threshold=0.5, average="micro", name="f1_score"),
        ],
    )
    return model


def hyperparameter_tuning(
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    validation_split: float,
    input_shape: tuple,
    patience: int,
    experiment_path: str,
    hyperparameter_tuning_epochs: int,
):
    # Patience
    patience_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    tuner = keras_tuner.Hyperband(
        lambda hp: build_model(hp, input_shape),
        objective=keras_tuner.Objective("val_f1_score", direction="max"),
        directory=experiment_path,
        project_name="tuning",
        overwrite=False,
    )

    tuner.search(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=hyperparameter_tuning_epochs,
        validation_split=validation_split,
        callbacks=[patience_callback],
    )
    logging.info("Hyperparameter tuning completed")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps


def run_experiment(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    input_shape: tuple,
    model_checkpoint: str,
    metrics_path: str,
    experiment_path: str,
    batch_size: int,
    epochs: int,
    validation_split: float,
    patience: int,
    hyperparameter_tuning_epochs: int,
):
    # Hyperparameter tuning
    logging.info("Starting hyperparameter tuning")
    best_hps = hyperparameter_tuning(
        x_train,
        y_train,
        batch_size,
        validation_split,
        input_shape,
        patience,
        experiment_path,
        hyperparameter_tuning_epochs,
    )

    # Complete training with the best hyperparameters
    logging.info("Training model with best hyperparameters [hps=%s]", best_hps.values)
    patience_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    model = build_model(best_hps, input_shape)
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[patience_callback],
    )

    # Evaluate model
    logging.info("Evaluating model on test set")
    test_loss, test_accuracy, test_precision, test_recall, test_auc, test_f1_score = (
        model.evaluate(x_test, y_test, verbose=0)
    )
    y_pred = model.predict(x_test, verbose=0).round()

    table = prettytable.PrettyTable()
    table.field_names = ["Metric", "Value"]
    table.add_row(["Test Loss", test_loss])
    table.add_row(["Test Accuracy", test_accuracy])
    table.add_row(["Test Precision", test_precision])
    table.add_row(["Test Recall", test_recall])
    table.add_row(["Test AUC", test_auc])
    table.add_row(["Test F1 Score", test_f1_score])
    print(table)

    logging.info("Generating training history and confusion matrix plots")
    # Plot training,validation history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["binary_accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_binary_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(metrics_path, "training_history.png"))
    plt.close()

    # Save metrics
    logging.info(f"Saving metrics to {metrics_path}")
    with open(os.path.join(metrics_path, "metrics.txt"), "w") as f:
        f.write(table.get_string())

    # Plot confusion matrix
    logging.info("Plotting confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(metrics_path, "confusion_matrix.png"))
    plt.close()

    # Save model
    logging.info(f"Saving model to {model_checkpoint}")
    model.save(model_checkpoint)


# FEATURES
# 0	amount	limitar(transaction.amount / max_amount)
# 1	installments	limitar(transaction.installments / max_installments)
# 2	amount_vs_avg	limitar((transaction.amount / customer.avg_amount) / amount_vs_avg_ratio)
# 3	hour_of_day	hora(transaction.requested_at) / 23 (0-23, UTC)
# 4	day_of_week	dia_da_semana(transaction.requested_at) / 6 (seg=0, dom=6)
# 5	minutes_since_last_tx	limitar(minutos / max_minutes) ou -1 se last_transaction: null
# 6	km_from_last_tx	limitar(last_transaction.km_from_current / max_km) ou -1 se last_transaction: null
# 7	km_from_home	limitar(terminal.km_from_home / max_km)
# 8	tx_count_24h	limitar(customer.tx_count_24h / max_tx_count_24h)
# 9	is_online	1 se terminal.is_online, senão 0
# 10	card_present	1 se terminal.card_present, senão 0
# 11	unknown_merchant	1 se merchant.id não estiver em customer.known_merchants, senão 0 (invertido: 1 = desconhecido)
# 12	mcc_risk	mcc_risk.json[merchant.mcc] (valor padrão 0.5)
# 13	merchant_avg_amount	limitar(merchant.avg_amount / max_merchant_avg_amount)


def plot_metrics_about_dataset(x: np.ndarray, y: np.ndarray, metrics_path: str):
    logging.info("Plotting metrics about the dataset")
    # Plot class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.savefig(os.path.join(metrics_path, "class_distribution.png"))
    plt.close()

    # Plot correlation matrix just for non categorical features (0-8)
    x_non_categorical = x[:, :9]  # Assuming first 9 features are non-categorical
    correlation_matrix = pd.DataFrame(x_non_categorical).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Matrix (Non-Categorical Features)")
    plt.savefig(os.path.join(metrics_path, "correlation_matrix.png"))
    plt.close()

    return correlation_matrix


def load_and_split_data(test_size: float, random_state: int, metrics_path: str):
    x, y = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Dataset info
    table = prettytable.PrettyTable()
    table.field_names = ["Split", "Count", "Percentage"]
    table.add_row(["Train", len(x_train), f"{len(x_train)/len(x)*100:.2f}%"])
    table.add_row(["Test", len(x_test), f"{len(x_test)/len(x)*100:.2f}%"])
    print(table)

    # Plot class distribution in train and test sets in the same plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y_train, label="Train", color="blue", alpha=0.5)
    sns.countplot(x=y_test, label="Test", color="orange", alpha=0.5)
    plt.title("Class Distribution in Train and Test Sets")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(os.path.join(metrics_path, "class_distribution_train_test.png"))
    plt.close()
    return x, y, (x_train, y_train, x_test, y_test)


def main(args: argparse.Namespace):
    logging.info(f"Experiment parameters: {args}")

    # Creating experiment directory
    experiment_path, model_checkpoint, metrics_path = create_experiment_directory(
        args.experiment_path
    )

    # Split train test
    x, y, (x_train, y_train, x_test, y_test) = load_and_split_data(
        test_size=args.test_split, random_state=args.seed, metrics_path=metrics_path
    )

    # Plot metrics about the dataset
    correlation_matrix = plot_metrics_about_dataset(x, y, metrics_path)

    # Based in the correlation matrix, reduce the dimensionality of the dataset by removing features with low correlation with the target variable (label) and \
    # high correlation with other features.
    if args.remove_correlated_features:
        logging.info("Removing correlated features based on correlation matrix")
        features_to_remove = []
        for i in range(correlation_matrix.shape[0]):
            if abs(correlation_matrix.iloc[i, -1]) < 0.1:  # Low correlation with label
                for j in range(correlation_matrix.shape[0]):
                    if (
                        i != j and abs(correlation_matrix.iloc[i, j]) > 0.8
                    ):  # High correlation with another feature
                        features_to_remove.append(i)
                        break
        logging.info(
            f"Removing features with low correlation with label and high correlation with other features: {features_to_remove}"
        )
        x_train = np.delete(x_train, features_to_remove, axis=1)
        x_test = np.delete(x_test, features_to_remove, axis=1)

    # Run experiment
    if not args.only_plot_info:
        run_experiment(
            x_train,
            y_train,
            x_test,
            y_test,
            input_shape=(x_train.shape[1],),
            model_checkpoint=model_checkpoint,
            metrics_path=metrics_path,
            experiment_path=experiment_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=args.validation_split,
            patience=args.patience,
            hyperparameter_tuning_epochs=args.tuning_epochs,
        )

    logging.info("Experiment completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS, help="Number of epochs to train"
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=VALIDATION_SPLIT,
        help="Proportion of the dataset to include in the validation split",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=TEST_SPLIT,
        help="Proportion of the dataset to include in the test split",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
        help="Number of epochs to wait for improvement before stopping training",
    )
    parser.add_argument(
        "--tuning_epochs",
        type=int,
        default=DEFAULT_TUNING_EPOCHS,
        help="Number of epochs to use for hyperparameter tuning",
    )
    parser.add_argument(
        "--only-plot-info",
        action="store_true",
        default=False,
        help="Only plot information about the dataset",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--remove-correlated-features",
        action="store_true",
        default=False,
        help="Remove features with low correlation with the target variable and high correlation with other features",
    )
    parser.add_argument(
        "--experiment-path",
        type=str,
        default=None,
        help="Path to the experiment directory",
    )
    args = parser.parse_args()
    main(args)
