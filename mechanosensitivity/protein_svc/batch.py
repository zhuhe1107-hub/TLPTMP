"""Ordered numbered-folder inference with a single model and bounded I/O prefetch."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from functools import lru_cache
import os
from pathlib import Path
import queue
import re
import tempfile
import threading
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .data import FEATURE_DIM, load_feature
from .model import load_model

HEADERS = ['Protein name', 'Species', 'Mutation site', 'Prediction']
LABELS = ('less function', 'more function', 'WT-like')


@lru_cache(maxsize=8)
def _name_pattern(codes):
    alternatives = '|'.join(re.escape(code) for code in sorted(codes, key=len, reverse=True))
    return re.compile(r'^(.+)_(' + alternatives + r')_?(.+)$')


def parse_name(name, species_codes):
    match = _name_pattern(tuple(species_codes)).fullmatch(Path(name).stem)
    if match is None:
        raise ValueError(f'{name}: expected PROTEIN_SPECIESmutation.npy; check species_codes.json.')
    protein, code, mutation = match.groups()
    return protein, species_codes[code], mutation


def find_folders(input_root, start_folder=1, end_folder=None):
    if start_folder < 0 or (end_folder is not None and end_folder < start_folder):
        raise ValueError('Invalid numeric folder range.')
    folders = {}
    with os.scandir(input_root) as entries:
        for entry in entries:
            if not entry.name.isascii() or not entry.name.isdigit() or not entry.is_dir():
                continue
            number = int(entry.name)
            if number in folders:
                raise ValueError(f'Duplicate numeric folder ID: {folders[number].name} and {entry.name}.')
            folders[number] = Path(entry.path)
    selected = [(number, path) for number, path in sorted(folders.items())
                if number >= start_folder and (end_folder is None or number <= end_folder)]
    if not selected:
        raise ValueError('No numeric input folders in the selected range.')
    return selected


def _folder_files(folder, species_codes):
    records = []
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.name.lower().endswith('.npy') and entry.is_file():
                metadata = parse_name(entry.name, species_codes)
                positions = tuple(int(value) for value in re.findall(r'\d+', metadata[2]))
                records.append((metadata[:2] + (positions, metadata[2], entry.name),
                                Path(entry.path), metadata))
    records.sort(key=lambda record: record[0])
    if not records:
        raise ValueError(f'{folder.name}: no .npy files directly inside this folder.')
    return records


def _read_row(matrix, index, path):
    try:
        matrix[index] = load_feature(path)
    except Exception as error:
        raise ValueError(f'{path}: {error}') from error


def _produce(folders, species_codes, batch_size, io_workers, pending, stop):
    def put(event):
        while not stop.is_set():
            try:
                pending.put(event, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    executor = ThreadPoolExecutor(max_workers=io_workers) if io_workers > 1 else None
    try:
        for number, folder in folders:
            if stop.is_set():
                return
            records = _folder_files(folder, species_codes)
            if not put(('start', number, len(records))):
                return
            for offset in range(0, len(records), batch_size):
                if stop.is_set():
                    return
                part = records[offset:offset + batch_size]
                matrix = np.empty((len(part), FEATURE_DIM), dtype=np.float32)
                if executor is None:
                    for index, record in enumerate(part):
                        _read_row(matrix, index, record[1])
                else:
                    futures = [executor.submit(_read_row, matrix, index, record[1])
                               for index, record in enumerate(part)]
                    for future in futures:
                        future.result()
                if not put(('batch', number, (matrix, [record[2] for record in part]))):
                    return
            if not put(('end', number, None)):
                return
        put(('done', None, None))
    except Exception as error:
        put(('error', None, error))
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


def run(input_root, output_dir, model_path, species_codes, *, batch_size=1024,
        prefetch_batches=2, io_workers=1, blas_threads=1, start_folder=1, end_folder=None):
    """Predict each selected folder; return run statistics. Never fit or reload the model."""
    for name, value in [('batch_size', batch_size), ('prefetch_batches', prefetch_batches),
                        ('io_workers', io_workers), ('blas_threads', blas_threads)]:
        if not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if not species_codes or any(not isinstance(key, str) or not key or
                                not isinstance(value, str) or not value
                                for key, value in species_codes.items()):
        raise ValueError('Species map must contain nonempty code/name strings.')
    folders = find_folders(input_root, start_folder, end_folder)
    output_dir = Path(output_dir)
    width = max(4, len(str(folders[-1][0])))
    destinations = {number: output_dir / f'{number:0{width}d}.csv' for number, _ in folders}
    remaining, skipped = [], []
    for number, folder in folders:
        destination = destinations[number]
        if destination.is_file():
            # Only a fully completed folder is published as a final CSV by this module.
            skipped.append(number)
            print(f'[SKIP] Completed output: {destination}', flush=True)
        elif destination.exists():
            raise FileExistsError(f'{destination} exists but is not a CSV file.')
        else:
            remaining.append((number, folder))
    folders = remaining
    if not folders:
        print('All selected folders already have results; nothing to predict.', flush=True)
        return {'folder_order': [], 'skipped_folders': skipped, 'n_samples': 0,
                'counts': {label: 0 for label in LABELS}, 'model_loads': 0,
                'model_load_seconds': 0.0, 'elapsed_seconds': 0.0, 'samples_per_second': 0.0,
                'batch_size': batch_size, 'prefetch_batches': prefetch_batches,
                'io_workers': io_workers, 'blas_threads': blas_threads, 'folders': []}
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = load_model(model_path)
    model_load_seconds = time.perf_counter() - started
    pending = queue.Queue(maxsize=prefetch_batches)
    stop = threading.Event()
    producer = threading.Thread(target=_produce,
                                args=(folders, species_codes, batch_size, io_workers, pending, stop),
                                name='npy-prefetch')
    totals = Counter({label: 0 for label in LABELS})
    folder_reports = []
    handle = None
    producer.start()
    try:
        with threadpool_limits(limits=blas_threads):
            while True:
                event, number, payload = pending.get()
                if event == 'error':
                    raise payload
                if event == 'done':
                    break
                if event == 'start':
                    handle = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8-sig', newline='',
                                                         dir=output_dir, prefix=f'.{number:0{width}d}.',
                                                         suffix='.partial', delete=False)
                    writer = csv.writer(handle)
                    writer.writerow(HEADERS)
                    count, expected = 0, payload
                    counts = Counter({label: 0 for label in LABELS})
                elif event == 'batch':
                    matrix, metadata = payload
                    labels = [LABELS[int(value)] for value in model.predict(matrix)]
                    if len(labels) != len(metadata):
                        raise ValueError('Model returned a different number of predictions than input rows.')
                    writer.writerows((*row, label) for row, label in zip(metadata, labels))
                    count += len(labels)
                    counts.update(labels)
                    # Do not retain the consumed matrix while waiting for the next batch.
                    del matrix, payload
                elif event == 'end':
                    if count != expected:
                        raise ValueError(f'{number}: incomplete folder, wrote {count}/{expected} rows.')
                    partial = Path(handle.name)
                    handle.close()
                    handle = None
                    # Publish atomically without replacing another run's completed CSV.
                    if os.name == 'nt':
                        partial.rename(destinations[number])
                    else:
                        os.link(partial, destinations[number])
                        partial.unlink()
                    totals.update(counts)
                    folder_reports.append({'folder': number, 'csv': destinations[number].name,
                                           'n_samples': count, 'counts': dict(counts)})
                    print(f'[{len(folder_reports)}/{len(folders)}] {destinations[number].name}: '
                          f'{count} rows, {dict(counts)}', flush=True)
    finally:
        stop.set()
        if handle is not None:
            handle.close()
        producer.join()
    elapsed = time.perf_counter() - started
    return {'folder_order': [number for number, _ in folders], 'skipped_folders': skipped,
            'n_samples': sum(totals.values()),
            'counts': dict(totals), 'model_loads': 1, 'model_load_seconds': model_load_seconds,
            'elapsed_seconds': elapsed, 'samples_per_second': sum(totals.values()) / elapsed,
            'batch_size': batch_size, 'prefetch_batches': prefetch_batches,
            'io_workers': io_workers, 'blas_threads': blas_threads, 'folders': folder_reports}
