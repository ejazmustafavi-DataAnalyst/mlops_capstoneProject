# src/api/utils.py
import joblib
import os
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'model', 'classifier.joblib')

_model_data = None

def load_model():
    global _model_data
    if _model_data is None:
        _model_data = joblib.load(MODEL_PATH)
    return _model_data

def predict_from_dict(feature_dict):
    data = load_model()
    model = data['model']
    feature_names = list(data['feature_names'])
    # Ensure all required features provided
    X = [feature_dict[name] for name in feature_names]
    X = np.array(X).reshape(1, -1)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0].tolist()
    return int(pred), proba

def predict_from_array(features_list):
    data = load_model()
    model = data['model']
    X = np.array(features_list).reshape(1, -1)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0].tolist()
    return int(pred), proba
