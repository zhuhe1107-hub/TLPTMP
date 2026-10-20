"""Read validated 3840-D feature files and write prediction results."""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

CLASSES = ['less', 'more', 'WT-like']
FEATURE_DIM = 3840


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def as_feature_matrix(values):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIM or values.shape[0] == 0:
        raise ValueError('Features must have shape (3840,) or (N,3840), N>=1.')
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError('Features must have floating-point dtype.')
    with np.errstate(over='ignore', invalid='ignore'):
        values = values.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError('Features contain NaN/Inf or overflow when converted to float32.')
    return values


def load_feature(path, expected_sha256=None):
    path = Path(path)
    raw = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f'{path.name}: feature SHA-256 mismatch.')
    values = as_feature_matrix(np.load(io.BytesIO(raw), allow_pickle=False))
    if values.shape[0] != 1:
        raise ValueError(f'{path.name}: each feature file must contain one sample.')
    return values[0]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot write a prediction table without samples.')
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
