# Thermosensitivity prediction

Predict TRPV1 thermal function and half-maximal response temperature (T50) from protein embeddings. Each sample receives a functional class (`functional` or `less functional`) and a temperature estimate in degrees Celsius.

## Research background

This component uses transfer learning based on pretrained OmegaTherm representations to study mutation-associated changes in temperature-sensitive ion channel function. Its curated thermosensitivity dataset contains wild-type and mutant sequences from TRPV1 and related ion channel proteins, with variants labeled as `functional` or `less functional`. Embeddings extracted using pretrained OmegaTherm weights are used to train downstream phenotype predictors. The resulting predictions help characterize mutation-associated thermal-response phenotypes and support further evaluation of selected variants through wet-lab experiments.

The upstream representation models and feature-extraction code are available in [OmegaTherm](https://github.com/CSUBioGroup/OmegaTherm). This component is part of [TLPTMP](https://github.com/maroonlove/TLPTMP), the transfer learning framework for predicting protein thermosensitivity and mechanosensitivity.

## Input

Store each sample as a one-dimensional floating-point NumPy `.npy` vector generated with the OmegaTherm representation and feature-construction method used by the predictor.

| Sample type | Vector shape | Example filename |
|---|---|---|
| Wild type | `(1280,)` | `TRPV1_MOUSE_WT.npy` |
| Mutant | `(3840,)` | `TRPV1_MOUSE_G644N.npy` |

Filenames follow `TRPV1_<SPECIES>_<MUTATION>.npy`. Supported species codes are `MOUSE`, `RAT`, `HUMAN`, and `AXOLOTL`. Directory prediction recursively selects TRPV1 NPY files. Protein, species, mutation, and sample names in the results are derived from these filenames.

## Installation

Run commands from the `thermosensitivity` directory:

```bash
conda create -n tlptmp-thermal python=3.13 -y
conda activate tlptmp-thermal
python -m pip install -r requirements.txt
```

The models use scikit-learn 1.7.2. Prediction runs on CPU.

## Predict

```bash
# Single sample
python predict.py --feature-file /path/to/TRPV1_MOUSE_G644N.npy --out predictions/single.csv

# Directory of TRPV1 embeddings
python predict.py --features-dir /path/to/TRPV1_embeddings --out predictions/results.csv

# Set the inference batch size
python predict.py --features-dir /path/to/TRPV1_embeddings --out predictions/batched.csv --batch-size 256
```

Both models are loaded once per process. The predictor encodes each feature vector and evaluates samples in batches. The output path must be new; a final CSV is published after inference completes. On Linux, the output filesystem must support hard links.

The output CSV contains:

| Column | Description |
|---|---|
| `protein` | Protein name |
| `species` | Species code |
| `mutation` | Mutation identifier |
| `predicted_label` | `functional` or `less functional` |
| `predicted_temperature_celsius` | Predicted half-maximal response temperature (T50), in degrees Celsius |
| `name` | Input filename without the `.npy` extension |

## Python API

```python
import numpy as np
from thermal_predictor import load_models, encode_features, predict_array

models = load_models("models")
embedding = np.load("/path/to/TRPV1_MOUSE_G644N.npy", allow_pickle=False)
features = encode_features([embedding])
labels, temperatures = predict_array(models, features)
```

Pass a list of feature vectors to `encode_features` for batch inference. It constructs a 5122-dimensional input with separate WT and mutant blocks and two presence flags. Load the models once and reuse them for subsequent batches. Keep both `.joblib` files and `model_info.json` in the `models/` directory.

## Models

| Task | Pipeline | Decision rule |
|---|---|---|
| Functional phenotype | StandardScaler -> RBF KernelPCA(48) -> RBF SVC, `C=1` | Apply the classifier's saved class-1 probability threshold of approximately 0.76; class 1 is `less functional` |
| T50 | StandardScaler -> PCA(8) -> RBF SVR, `C=80`, `epsilon=0.5` | Return the regression output in degrees Celsius |

Use the prediction script or `predict_array` to apply the saved classification threshold. Functional annotations and temperature measurements are available in `data/`.
