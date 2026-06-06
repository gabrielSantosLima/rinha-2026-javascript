import express from "express";
import cors from "cors";
import { NormalizeUseCase } from "./use_cases/normalize_use_case.js";
import { DetectFraudUseCase } from "./use_cases/detect_fraud_use_case.js";

const app = express();
const normalizeUseCase = new NormalizeUseCase();
const detectFraudUseCase = new DetectFraudUseCase();

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cors());

await detectFraudUseCase.initializeModel();

app.get("/ready", async (req, res) => {
  return res.status(200);
});

app.post("/fraud-score", async (req, res) => {
  const content = req.body;
  const transaction = normalizeUseCase.execute(content);
  console.log(transaction);
  const fraudDetectionResponse = await detectFraudUseCase.execute(transaction);
  return res.json(fraudDetectionResponse);
});

app.listen(3333, () => {
  console.log("API running on http://localhost:3333");
});
