# src/api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from typing import Dict, Any
from .schemas import NamedFeatures, ArrayFeatures, PredictionResponse
from .utils import load_model, predict_from_dict, predict_from_array
import os

app = FastAPI(
    title="MLOps Capstone - Breast Cancer Predictor",
    version="0.1.0",
    description="Predict malignant(0)/benign(1) from 30 numeric features using a saved sklearn pipeline"
)

@app.on_event("startup")
def startup_event():
    # load model once
    load_model()

@app.get("/")
def root():
    return {"message": "MLOps Capstone API", "status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: Dict[str, Any]):
    """
    Accepts either:
    1) JSON object with named features: { "mean radius": 12.3, "mean texture": 14.5, ... }
    2) JSON with ordered features: { "features": [f1, f2, ..., f30] }
    """
    # Try array-style first
    if "features" in payload:
        try:
            arr = ArrayFeatures(**payload)
            features_list = arr.features
            pred, proba = predict_from_array(features_list)
            return {"predicted_label": pred, "predicted_proba": proba}
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))

    # Otherwise assume named dict
    try:
        nf = NamedFeatures(__root__=payload)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Validate numeric values
    feature_dict = {}
    for k, v in nf.__root__.items():
        try:
            feature_dict[k] = float(v)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Feature '{k}' must be numeric")

    # Check required features exist
    model_meta = load_model()
    required = set(model_meta['feature_names'])
    provided = set(feature_dict.keys())
    missing = required - provided
    extra = provided - required
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {sorted(list(missing))[:6]}{'...' if len(missing)>6 else ''}")
    # Optionally warn about extras (ignored)
    pred, proba = predict_from_dict(feature_dict)
    return {"predicted_label": pred, "predicted_proba": proba}
