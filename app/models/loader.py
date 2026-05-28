import joblib
from pathlib import Path
from typing import Any

MODELS_PATH = Path(__file__).parent

def load_model(model_name: str) -> Any:
    model_path = MODELS_PATH / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)

def get_maternal_health_model():
    return load_model("maternal_health_model")

def get_preeclampsia_model():
    return load_model("preeclampsia_model")