"""Inference from an integrity-checked saved PCA+SVC Pipeline."""
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .data import CLASSES, FEATURE_DIM, as_feature_matrix, sha256


def build_model_structure():
    """Declare PCA(3840 -> 20) followed by an RBF SVC, without fitting.

    This is the architecture declaration only. Inference uses the saved fitted
    Pipeline, including its PCA arrays, support vectors and class weights.
    """
    return Pipeline([
        ('pca', PCA(n_components=20, svd_solver='full', whiten=False)),
        ('svc', SVC(kernel='rbf', C=3.0, gamma='auto', probability=False,
                    random_state=42)),
    ])


def load_model(model_path):
    model_path = Path(model_path)
    metadata_path = model_path.parent / 'model_info.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if sha256(model_path) != metadata['model_sha256']:
        raise ValueError('Model SHA-256 does not match model_info.json.')
    if metadata['classes'] != CLASSES or metadata['feature_dim'] != FEATURE_DIM:
        raise ValueError('Model class mapping or feature dimension does not match this project.')
    if metadata['prediction_rule'] != 'native_pipeline_predict':
        raise ValueError('This project uses native SVC.predict only.')
    model = joblib.load(model_path)
    structure = build_model_structure()
    if (not isinstance(model, Pipeline) or
            list(model.named_steps) != list(structure.named_steps) or
            model.n_features_in_ != FEATURE_DIM):
        raise ValueError('Expected a 3840-D PCA+SVC Pipeline.')
    for name, component in structure.steps:
        if not isinstance(model.named_steps[name], type(component)):
            raise ValueError(f'Expected {type(component).__name__} in Pipeline step {name}.')
    np.testing.assert_array_equal(model.classes_, [0, 1, 2])
    return model


def predict_array(model, features):
    features = as_feature_matrix(features)
    with threadpool_limits(limits=1):
        prediction = model.predict(features)
    return np.asarray(CLASSES)[prediction.astype(int)]


def runtime_versions():
    return {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'scikit_learn': sklearn.__version__, 'joblib': joblib.__version__}
