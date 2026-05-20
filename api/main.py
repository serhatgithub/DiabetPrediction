import os
from typing import Literal

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


KSERVE_PREDICT_URL = os.getenv(
    "KSERVE_PREDICT_URL",
    "http://diabetes-predictor-predictor.diabet.svc.cluster.local/v1/models/diabetes-predictor:predict",
)


class DiabetesInput(BaseModel):
    Pregnancies: float = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    BloodPressure: float = Field(..., ge=0)
    SkinThickness: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    risk: Literal["low", "high"]
    model_server: str


app = FastAPI(
    title="Diabetes Prediction API",
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8" />
  <title>Diabetes Prediction</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #f4f6f8;
      margin: 0;
      padding: 40px;
    }

    .container {
      max-width: 760px;
      margin: auto;
      background: white;
      padding: 28px;
      border-radius: 14px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    }

    h1 {
      margin-top: 0;
      color: #222;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 14px;
    }

    label {
      display: block;
      font-size: 13px;
      color: #555;
      margin-bottom: 5px;
    }

    input {
      width: 100%;
      padding: 10px;
      border: 1px solid #ccc;
      border-radius: 8px;
      font-size: 14px;
      box-sizing: border-box;
    }

    button {
      margin-top: 22px;
      width: 100%;
      padding: 13px;
      background: #2563eb;
      color: white;
      border: none;
      border-radius: 10px;
      font-size: 16px;
      cursor: pointer;
    }

    button:hover {
      background: #1d4ed8;
    }

    .result {
      margin-top: 24px;
      padding: 16px;
      border-radius: 10px;
      display: none;
      font-size: 16px;
    }

    .low {
      background: #dcfce7;
      color: #166534;
    }

    .high {
      background: #fee2e2;
      color: #991b1b;
    }

    .error {
      background: #fef3c7;
      color: #92400e;
    }

    .small {
      font-size: 13px;
      color: #666;
      margin-top: 12px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Diabetes Prediction</h1>
    <p>Hasta değerlerini gir, model diyabet risk tahmini döndürsün.</p>

    <form id="predictForm">
      <div class="grid">
        <div>
          <label>Pregnancies</label>
          <input name="Pregnancies" type="number" step="any" value="6" required />
        </div>

        <div>
          <label>Glucose</label>
          <input name="Glucose" type="number" step="any" value="148" required />
        </div>

        <div>
          <label>Blood Pressure</label>
          <input name="BloodPressure" type="number" step="any" value="72" required />
        </div>

        <div>
          <label>Skin Thickness</label>
          <input name="SkinThickness" type="number" step="any" value="35" required />
        </div>

        <div>
          <label>Insulin</label>
          <input name="Insulin" type="number" step="any" value="0" required />
        </div>

        <div>
          <label>BMI</label>
          <input name="BMI" type="number" step="any" value="33.6" required />
        </div>

        <div>
          <label>Diabetes Pedigree Function</label>
          <input name="DiabetesPedigreeFunction" type="number" step="any" value="0.627" required />
        </div>

        <div>
          <label>Age</label>
          <input name="Age" type="number" step="any" value="50" required />
        </div>
      </div>

      <button type="submit">Predict</button>
    </form>

    <div id="result" class="result"></div>

    <div class="small">
      Backend: FastAPI → KServe → ML model
    </div>
  </div>

  <script>
    const form = document.getElementById("predictForm");
    const resultBox = document.getElementById("result");

    form.addEventListener("submit", async (event) => {
      event.preventDefault();

      resultBox.style.display = "block";
      resultBox.className = "result";
      resultBox.innerText = "Tahmin alınıyor...";

      const formData = new FormData(form);
      const payload = {};

      for (const [key, value] of formData.entries()) {
        payload[key] = Number(value);
      }

      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
          resultBox.className = "result error";
          resultBox.innerText = "Hata: " + JSON.stringify(data);
          return;
        }

        resultBox.className = "result " + data.risk;

        if (data.prediction === 1) {
          resultBox.innerText = "Prediction: 1 | Risk: HIGH";
        } else {
          resultBox.innerText = "Prediction: 0 | Risk: LOW";
        }
      } catch (error) {
        resultBox.className = "result error";
        resultBox.innerText = "İstek hatası: " + error;
      }
    });
  </script>
</body>
</html>
"""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: DiabetesInput):
    instance = [
        payload.Pregnancies,
        payload.Glucose,
        payload.BloodPressure,
        payload.SkinThickness,
        payload.Insulin,
        payload.BMI,
        payload.DiabetesPedigreeFunction,
        payload.Age,
    ]

    kserve_payload = {
        "instances": [
            instance
        ]
    }

    try:
        response = requests.post(
            KSERVE_PREDICT_URL,
            json=kserve_payload,
            timeout=10,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"KServe prediction service is unreachable: {exc}",
        )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "KServe prediction failed",
                "status_code": response.status_code,
                "response": response.text,
            },
        )

    result = response.json()
    predictions = result.get("predictions")

    if not predictions:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "KServe response does not contain predictions",
                "response": result,
            },
        )

    prediction = int(predictions[0])

    return PredictionResponse(
        prediction=prediction,
        risk="high" if prediction == 1 else "low",
        model_server="kserve",
    )
