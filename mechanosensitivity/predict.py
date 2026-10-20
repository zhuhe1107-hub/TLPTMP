"""Predict one feature file or a directory using an already trained Pipeline."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from protein_svc import load_model, predict_array
from protein_svc.data import CLASSES, load_feature, sha256, write_csv, write_json

ROOT = Path(__file__).resolve().parent


def run(model_path, features_dir, feature_file, pattern, out, batch_size):
    if batch_size < 1:
        raise ValueError('--batch-size must be positive.')
    summary_path = out.with_suffix('.summary.json')
    if out.is_file():
        if not summary_path.is_file():
            raise ValueError(f'{out}: CSV exists without a completion summary; verify it before reusing.')
        summary = json.loads(summary_path.read_text(encoding='utf-8-sig'))
        if summary.get('csv_sha256') != sha256(out):
            raise ValueError(f'{out}: existing CSV does not match its completion summary.')
        if summary.get('model_sha256') != sha256(model_path):
            raise ValueError(f'{out}: existing results belong to a different model; use a new --out.')
        print(f'[SKIP] Completed output: {out}', flush=True)
        return
    if out.exists() or summary_path.exists():
        raise FileExistsError('Output path or orphaned summary already exists; choose a new --out.')
    if feature_file is not None:
        parent = feature_file.resolve().parent
        paths = [feature_file.resolve()]
    else:
        parent = features_dir.resolve()
        paths = sorted(path for path in parent.rglob(pattern) if path.is_file())
    if not paths:
        raise ValueError('No feature files matched the input path/pattern.')
    model_hash = sha256(model_path)
    model = load_model(model_path)
    rows = []
    for start in range(0, len(paths), batch_size):
        batch = paths[start:start + batch_size]
        vectors = np.stack([load_feature(path) for path in batch])
        predictions = predict_array(model, vectors)
        rows.extend({'sample_id': path.stem, 'relative_file': path.relative_to(parent).as_posix(),
                     'prediction': str(label), 'feature_sha256': sha256(path)} for path, label in zip(batch, predictions))
    if sha256(model_path) != model_hash:
        raise ValueError('Model file changed during inference.')
    counts = Counter(row['prediction'] for row in rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, rows)
    summary = {'n_samples': len(rows), 'counts': {label: counts[label] for label in CLASSES},
               'fractions': {label: counts[label] / len(rows) for label in CLASSES},
               'all_three_classes_present': all(counts[label] > 0 for label in CLASSES),
               'model_sha256': model_hash, 'prediction_rule': 'native_pipeline_predict', 'retrained': False,
               'csv_sha256': sha256(out)}
    write_json(summary_path, summary)
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/svc_pca.joblib')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--features-dir', type=Path, help='Recursively read matching .npy files.')
    source.add_argument('--feature-file', type=Path, help='One (3840,) or (1,3840) .npy file.')
    parser.add_argument('--pattern', default='*.npy', help='Directory filename filter, e.g. TRPV4_HUMAN*.npy')
    parser.add_argument('--out', type=Path, required=True, help='New CSV output path.')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    run(args.model.resolve(), args.features_dir, args.feature_file, args.pattern, args.out.resolve(), args.batch_size)
