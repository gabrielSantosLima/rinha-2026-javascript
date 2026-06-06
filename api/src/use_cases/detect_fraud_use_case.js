import ort from "onnxruntime-node";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

export class DetectFraudUseCase {
  constructor() {
    this.session = null;
  }

  async initializeModel() {
    const modelPath = join(__dirname, "..", "..", "model.onnx");
    console.log(`Loading model from: ${modelPath}`);
    this.session = await ort.InferenceSession.create(modelPath);
    console.log("Model loaded successfully!");
  }

  async execute(transactionVector) {
    if (!this.session) {
      throw new Error("Model not initialized. Call initializeModel first.");
    }

    const rawData = new Float32Array(transactionVector);
    const inputTensor = new ort.Tensor("float32", rawData, [1, 14]);
    const feeds = { "serving_default_input_layer:0": inputTensor };
    const outputMap = await this.session.run(feeds);
    const outputTensor = outputMap["StatefulPartitionedCall_1:0"].cpuData[0];

    let probability = parseFloat(outputTensor);
    const isFraud = probability >= 0.5;
    if (!isFraud) probability = 1 - probability;

    console.log(`Model output: ${outputTensor}`);

    return {
      approved: !isFraud,
      fraud_score: probability,
    };
  }
}
