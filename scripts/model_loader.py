import os
import joblib
import logging

def load_models(base_path: str = "scripts/models"):
    """
    Loads all required ML models from the specified path and returns them as a dictionary.

    Args:
        base_path (str): Path to the folder containing the model .pkl files.

    Returns:
        dict: A dictionary with model names as keys and loaded model objects as values.
    """
    model_files = {
        "spread_model": "spread_model.pkl",
        "inventory_model": "inventory_model.pkl",
        "trend_model": "trend_model.pkl",
        "volatility_model": "volatility_model.pkl"
    }

    models = {}
    for name, filename in model_files.items():
        try:
            path = os.path.join(base_path, filename)
            models[name] = joblib.load(path)
            logging.info(f"Loaded model: {name} from {path}")
        except Exception as e:
            logging.error(f"Failed to load model {name} from {filename}: {e}")
            raise

    return models
