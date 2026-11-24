# PLATO LOPS2 classification using features

A simple library to extract time and frequency domain features from preprocessed TESS light curves meant to be run in the IvS Pleaids cluster. The classification results obtained from the feature-based ML classifier are part of [(Kliapets et al. 2026)](https://arxiv.org/abs/2604.12758).

To download the LOPS2 variability classification catalogue please see: https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/GENZHD

Dependencies are managed with [uv](https://docs.astral.sh/uv/getting-started/installation/)

Install the dependencies with

```bash
uv sync
```

Prepare light curve parquets for the training set with

```bash
uv run src/runner.py preparation ../plato_fov/training_df_try8.parquet results --n-jobs=16
```

Then compute the training set features with

```bash
uv run src/runner.py training results/light_curves/training_df_try8/ results/ --n-jobs=16
```

Prepare light curve parquets for all sectors with

```
uv run src/runner.py preparation ../plato_fov/paths/ results --n-jobs=16
```

and extract features with

```bash
uv run src/runner.py inference results/light_curves/s0001 results --n-jobs=16
```
You can also pass `results/light_curves/` to compute in all sectors

Notes:

- `nifty_ls` is used to compute Lomb-Scargle periodograms. This library fails on nodes with CPUs missing avx2 instructions (pleiad02 for example). polars also have issues with these nodes, but this can be solved installing `polars-lts-cpu`.
- Export `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` to avoid CPU oversubscription
