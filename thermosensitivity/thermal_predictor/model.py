"""Explicit thermal model architectures and inference from fitted artifacts."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.decomposition import KernelPCA, PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from threadpoolctl import threadpool_limits

FEATURE_DIM = 5122


def build_model_structures():
    """Declare the two unfitted structures; inference loads the saved state."""
    return {
        'functional': Pipeline([
            ('standardscaler', StandardScaler()),
            ('kernelpca', KernelPCA(n_components=48, kernel='rbf', n_jobs=-1, random_state=42)),
            ('svc', SVC(C=1, kernel='rbf', class_weight='balanced', probability=True, random_state=42)),
        ]),
        'temperature': Pipeline([
            ('standardscaler', StandardScaler()),
            ('pca', PCA(n_components=8, random_state=42)),
            ('svr', SVR(C=80, epsilon=0.5, kernel='rbf')),
        ]),
    }


def load_models(model_dir):
    """Load each saved artifact once; do not replace or fit pipeline steps."""
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / 'model_info.json').read_text(encoding='utf-8'))
    expected_version = metadata['serialized_sklearn_version']
    if sklearn.__version__ != expected_version:
        raise RuntimeError(f'Thermal checkpoints require scikit-learn {expected_version}; found {sklearn.__version__}. Use the thermal environment.')
    structures = build_model_structures()
    models = {}
    for role in ['functional', 'temperature']:
        info = metadata[role]
        path = model_dir / info['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'{path.name}: model SHA-256 mismatch.')
        artifact = joblib.load(path)
        estimator = artifact['estimator']
        expected = structures[role]
        if (not isinstance(estimator, Pipeline) or
                list(estimator.named_steps) != list(expected.named_steps) or
                estimator.n_features_in_ != FEATURE_DIM):
            raise ValueError(f'{path.name}: unexpected pipeline or input dimension.')
        for name, step in expected.steps:
            if not isinstance(estimator.named_steps[name], type(step)):
                raise ValueError(f'{path.name}: expected {type(step).__name__} in {name}.')
        models[role] = artifact
    np.testing.assert_array_equal(models['functional']['estimator'].classes_, [0, 1])
    if models['functional']['labels'] != {0: 'functional', 1: 'less functional'}:
        raise ValueError('Unexpected thermal class mapping.')
    if models['temperature']['endpoint'] != 'halfmax':
        raise ValueError('Expected a half-maximal response temperature model.')
    return models


def encode_features(vectors):
    """Encode 1280-D WT or 3840-D mutant vectors in the original 5122-D layout."""
    if len(vectors) == 0:
        raise ValueError('At least one feature vector is required.')
    features = np.zeros((len(vectors), FEATURE_DIM), dtype=np.float32)
    for index, vector in enumerate(vectors):
        vector = np.asarray(vector)
        if vector.ndim != 1 or vector.size not in (1280, 3840):
            raise ValueError('Each thermal NPY must contain one vector with shape (1280,) or (3840,).')
        if not np.issubdtype(vector.dtype, np.floating):
            raise ValueError('Thermal features must have a floating-point dtype.')
        with np.errstate(over='ignore', invalid='ignore'):
            vector = vector.astype(np.float32, copy=False)
        if not np.isfinite(vector).all():
            raise ValueError('Thermal features contain NaN/Inf or float32 overflow.')
        if vector.size == 1280:
            features[index, :1280] = vector
            features[index, 5120] = 1.0
        else:
            features[index, 1280:5120] = vector
            features[index, 5121] = 1.0
    return features


def predict_array(models, features):
    """Apply the saved class-1 threshold and the saved temperature regressor."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM or not np.isfinite(features).all():
        raise ValueError('Expected finite encoded features with shape (N, 5122).')
    functional = models['functional']
    with threadpool_limits(limits=1):
        probabilities = functional['estimator'].predict_proba(features)[:, 1]
        indices = (probabilities >= functional['threshold']).astype(int)
        temperatures = models['temperature']['estimator'].predict(features)
    labels = np.asarray([functional['labels'][int(index)] for index in indices])
    return labels, np.asarray(temperatures, dtype=float)
