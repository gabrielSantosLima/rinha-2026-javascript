import os
import gzip
import json
from pathlib import Path

import pandas
import prettytable
import requests

DATASET_FEATURES_PATH = "https://github.com/zanfranceschi/rinha-de-backend-2026/raw/refs/heads/main/resources/references.json.gz"
DATASET_MCC_PATH = "https://github.com/zanfranceschi/rinha-de-backend-2026/raw/refs/heads/main/resources/mcc_risk.json"
DATASET_NORMALIZATION_PATH = "https://github.com/zanfranceschi/rinha-de-backend-2026/raw/refs/heads/main/resources/normalization.json"

DATASET_PATH = os.path.join(os.getcwd(), "data")

FEATURE_NAMES = [
    "amount",
    "installments",
    "amount_vs_avg",
    "hour_of_day",
    "day_of_week",
    "minutes_since_last_tx",
    "km_from_last_tx",
    "km_from_home",
    "tx_count_24h",
    "is_online",
    "card_present",
    "unknown_merchant",
    "mcc_risk",
    "merchant_avg_amount",
]

RESOURCES = {
    "features": (DATASET_FEATURES_PATH, "references.json.gz"),
    "mcc_risk": (DATASET_MCC_PATH, "mcc_risk.json"),
    "normalization": (DATASET_NORMALIZATION_PATH, "normalization.json"),
}


def download_file(url: str, output_path: Path) -> None:
    """Download a remote file unless it already exists locally."""
    if output_path.exists():
        print(f"Using cached file: {output_path}")
        return

    print(f"Downloading {url} -> {output_path}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with output_path.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)


def load_json(path: Path):
    """Load JSON from plain .json or gzipped .json.gz files."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as file:
        return json.load(file)


def normalize_records(raw_data):
    """Return a list of records regardless of the JSON top-level shape."""
    if isinstance(raw_data, list):
        return raw_data

    if isinstance(raw_data, dict):
        for key in ("data", "features", "references", "rows", "items"):
            value = raw_data.get(key)
            if isinstance(value, list):
                return value

    raise ValueError("Could not find dataset records in references.json.gz")


def build_features_dataframe(records) -> pandas.DataFrame:
    """Convert downloaded records into a pandas DataFrame with documented feature names."""
    dataset = pandas.DataFrame(records)

    if len(dataset.columns) == len(FEATURE_NAMES):
        dataset.columns = FEATURE_NAMES
    elif all(index in dataset.columns for index in range(len(FEATURE_NAMES))):
        dataset = dataset.rename(columns=dict(enumerate(FEATURE_NAMES)))
    elif "features" in dataset.columns:
        features = pandas.DataFrame(dataset["features"].tolist(), columns=FEATURE_NAMES)
        other_columns = dataset.drop(columns=["features"])
        dataset = pandas.concat([features, other_columns], axis=1)

    return dataset


def print_table(title: str, rows: list[tuple[str, object]]) -> None:
    table = prettytable.PrettyTable()
    table.title = title
    table.field_names = ["Metric", "Value"]
    table.align["Metric"] = "l"
    table.align["Value"] = "r"

    for metric, value in rows:
        table.add_row([metric, value])

    print(table)


def main() -> None:
    dataset_dir = Path(DATASET_PATH)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files = {}
    for resource_name, (url, filename) in RESOURCES.items():
        output_path = dataset_dir / filename
        download_file(url, output_path)
        downloaded_files[resource_name] = output_path

    raw_records = load_json(downloaded_files["features"])
    records = normalize_records(raw_records)
    dataset = build_features_dataframe(records)

    dataset_path = dataset_dir / "references.csv"
    dataset.to_csv(dataset_path, index=False)

    mcc_risk = load_json(downloaded_files["mcc_risk"])
    normalization = load_json(downloaded_files["normalization"])

    print_table(
        "Downloaded Resources",
        [
            (name, f"{path.name} ({path.stat().st_size / 1024 / 1024:.2f} MB)")
            for name, path in downloaded_files.items()
        ],
    )

    print_table(
        "Features Dataset",
        [
            ("Rows", f"{len(dataset):,}"),
            ("Columns", len(dataset.columns)),
            ("Memory", f"{dataset.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"),
            ("Output", dataset_path),
        ],
    )

    print_table(
        "Reference Data",
        [
            ("MCC risk entries", len(mcc_risk)),
            ("Normalization fields", len(normalization)),
        ],
    )

    print("\nDataset columns:")
    columns_table = prettytable.PrettyTable()
    columns_table.field_names = ["#", "Feature"]
    for index, name in enumerate(dataset.columns):
        columns_table.add_row([index, name])
    print(columns_table)


if __name__ == "__main__":
    main()




