import os
import tensorflow as tf
import pandas as pd
import numpy as np
import keras_tuner
import seaborn as sns
import prettytable
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from datetime import datetime
from tensorflow import keras
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

CURRENT_PATH = os.getcwd()

# Experiment setup
experiment_time = datetime.now().strftime("%Y%m%d-%H%M%S")
EXPERIMENT_PATH = os.path.join(CURRENT_PATH, f"experiment_{experiment_time}")
MODEL_CHECKPOINT = os.path.join(EXPERIMENT_PATH, "model.keras")
METRICS_PATH = os.path.join(EXPERIMENT_PATH, "metrics")

# Dataset setup
DATASET_PATH = os.path.join(CURRENT_PATH, "data", "references.csv")

# Training setup
BATCH_SIZE = 128
EPOCHS = 50
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1


def create_experiment_directory():
    if not os.path.exists(EXPERIMENT_PATH):
        os.makedirs(EXPERIMENT_PATH)
    if not os.path.exists(METRICS_PATH):
        os.makedirs(METRICS_PATH)


def load_and_preprocess_data():
    data = pd.read_csv(DATASET_PATH)
    x = np.array(data["vector"].apply(eval).tolist())
    y = np.array(data["label"].apply(lambda x: 1 if x == "fraud" else 0).tolist())
    return x, y


def main():
    # Creating experiment directory
    create_experiment_directory()

    # Load and preprocess your dataset here
    x, y = load_and_preprocess_data()

    # Split train test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SPLIT, random_state=42
    )

    # Define your model architecture here
    model = keras.Sequential(
        [
            keras.layers.Dense(128, activation="relu", input_shape=(x.shape[1],)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Compile and train your model here
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
    )

    # Save the model
    model.save(MODEL_CHECKPOINT)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    # Print the metrics using prettytable. Plot accuracy, precision, recall and f1-score using seaborn.
    table = prettytable.PrettyTable()
    table.field_names = ["Metric", "Value"]
    table.add_row(["Test Loss", test_loss])
    table.add_row(["Test Accuracy", test_accuracy])
    table.add_row(["Precision", precision_score(y_test, model.predict(x_test).round())])
    table.add_row(["Recall", recall_score(y_test, model.predict(x_test).round())])
    table.add_row(["F1 Score", f1_score(y_test, model.predict(x_test).round())])
    print(table)

    # Save metrics to a text file
    with open(os.path.join(METRICS_PATH, "metrics.txt"), "w") as f:
        f.write(table.get_string())

    # Plot confusion matrix
    y_pred = model.predict(x_test).round()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(METRICS_PATH, "confusion_matrix.png"))
    plt.close()


if __name__ == "__main__":
    main()
