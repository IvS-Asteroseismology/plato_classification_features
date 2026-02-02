# PLATO classification using features

A simple library to extract time and frequency domain features from preprocessed TESS light curves

Dependencies are managed with [uv](https://docs.astral.sh/uv/getting-started/installation/)

Install with 

```bash
uv sync
```

Feature extraction in training set

```bash
uv run src/runner.py training path_to_training_parquet results --n_jobs=8
```

Feature extraction in inferenec set

```bash
uv run src/runner.py inferenceg path_to_directory_with_parquets results --n_jobs=8
```
