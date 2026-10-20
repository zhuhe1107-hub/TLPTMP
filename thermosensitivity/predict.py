"""Predict TRPV1 thermal phenotype and half-maximal response temperature."""
import argparse
import csv
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from thermal_predictor import encode_features, load_models, predict_array

ROOT = Path(__file__).resolve().parent
COLUMNS = ['protein', 'species', 'mutation', 'predicted_label',
           'predicted_temperature_celsius', 'name']


def parse_filename(path):
    stem = Path(path).stem
    match = re.fullmatch(r'(TRPV1)_(MOUSE|RAT|HUMAN|AXOLOTL)[_-]?(.*)', stem, re.IGNORECASE)
    if match:
        protein, species, mutation = match.groups()
        return protein.upper(), species.upper(), mutation
    return 'TRPV1', 'UNKNOWN', stem


def run(model_dir, feature_file, features_dir, out, batch_size):
    if batch_size < 1:
        raise ValueError('--batch-size must be positive.')
    if out.exists():
        raise FileExistsError('Output already exists; choose a new --out.')
    paths = ([feature_file.resolve()] if feature_file is not None else
             sorted(path for path in features_dir.resolve().rglob('*.npy')
                    if path.is_file() and path.stem.upper().startswith('TRPV1')))
    if not paths or any(not path.stem.upper().startswith('TRPV1') for path in paths):
        raise ValueError('No TRPV1 NPY files matched the input; filenames must start with TRPV1.')
    models = load_models(model_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Keep failed runs separate from completed output files.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8-sig', newline='',
                                     dir=out.parent, prefix='.' + out.name + '.',
                                     suffix='.partial', delete=False) as handle:
        partial = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for start in range(0, len(paths), batch_size):
            batch = paths[start:start + batch_size]
            features = encode_features([np.load(path, allow_pickle=False) for path in batch])
            labels, temperatures = predict_array(models, features)
            for path, label, temperature in zip(batch, labels, temperatures):
                protein, species, mutation = parse_filename(path)
                writer.writerow({'protein': protein, 'species': species, 'mutation': mutation,
                                 'predicted_label': str(label),
                                 'predicted_temperature_celsius': float(temperature),
                                 'name': path.stem})
    if os.name == 'nt':
        partial.rename(out)
    else:
        os.link(partial, out)
        partial.unlink()
    print(f'Predicted {len(paths)} samples; saved {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--feature-file', type=Path)
    source.add_argument('--features-dir', type=Path)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    run(args.model_dir.resolve(), args.feature_file, args.features_dir, args.out.resolve(), args.batch_size)


if __name__ == '__main__':
    main()
