import tensorflow
import argparse
import os
import logging
from custom_metrics import BinaryFBetaScore
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={os.environ.get('CONDA_PREFIX')}"
CURRENT_PATH = os.getcwd()


def main(args: argparse.Namespace):
    logging.info(f"Starting model conversion [args={args}]")

    # Load the model
    logging.info(f"Loading model from {args.model}...")
    output_name = f"{os.path.basename(args.model)}.tflite"
    output_path = os.path.join(CURRENT_PATH, output_name)
    logging.info(f"Model loaded successfully. Output path will be {output_path}")

    assert os.path.exists(
        args.model
    ), "Failed to load the model. Please check the path and try again."
    assert not os.path.exists(
        output_path
    ), f"Output file already exists at {output_path}. Please remove the existing file or choose a different output path."

    # Convert to Tflite
    logging.info("Converting model to TFLite format...")
    model = tensorflow.keras.models.load_model(
        args.model, custom_objects={"BinaryFBetaScore": BinaryFBetaScore}
    )
    converter = tensorflow.lite.TFLiteConverter.from_keras_model(model)

    # Quantization
    converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tensorflow.float16]

    # Converting...
    logging.info("Converting model to TFLite format...")
    tflite_model = converter.convert()

    # Save the model
    logging.info(f"Saving converted model to {output_path}...")
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    logging.info(
        f"Model conversion completed successfully. Output saved to {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to the model to convert")
    args = parser.parse_args()
    main(args)
