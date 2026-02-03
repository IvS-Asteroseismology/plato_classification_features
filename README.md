# PLATO classification using features

A simple library to extract time and frequency domain features from preprocessed TESS light curves meant to be run in the IvS Pleaids cluster.

Dependencies are managed with [uv](https://docs.astral.sh/uv/getting-started/installation/)

Install the dependencies with

```bash
uv sync
```

Feature extraction in training set

```bash
uv run src/runner.py training path_to_training_parquet results --n-jobs=8
```

Feature extraction in inference set

```bash
uv run src/runner.py inference path_to_directory_with_parquets results --n-jobs=8
```

Notes:

- `nifty_ls` is used to compute Lomb-Scargle periodograms. This library fails on nodes with CPUs missing avx2 instructions (pleiad02 for example). polars also have issues with these nodes, but this can be solved installing `polars-lts-cpu`.
