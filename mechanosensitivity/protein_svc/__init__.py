"""Public inference API: load_model(path), predict_array(model, features)."""
from .model import load_model, predict_array

__all__ = ['load_model', 'predict_array']
