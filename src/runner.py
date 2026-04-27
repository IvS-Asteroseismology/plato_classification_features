from collections import Counter
from pathlib import Path
from functools import partial
from typing import Any, Callable
import argparse
import logging
import polars as pl

from parallel_utils import extract_in_parallel
from extractors import ExtractorFn, extract_features_from_slice, extract_light_curve_from_slice
from eta_estimator import ETAEstimator
from config import PipelineConfig


log = logging.getLogger(__name__)


def parquet_num_rows(parquet_path: str | Path) -> int:
    return pl.scan_parquet(parquet_path).select(pl.len()).collect().item()

def report_errors(errors: list[dict[str, Any]], total: int) -> None:
    if errors:
        log.error(f"{len(errors)}/{total} files failed")
        for reason, count in Counter(e["_error"] for e in errors).items():
            log.error(f"\t {reason}: {count}")

def normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    if "FilePath" in df.columns:
        return df.rename({"FilePath": "path"})
    return df
    
def process_parquet(
        parquet_path: Path,
        parallel_extractor: Callable,
        save_dir: Path, 
        overwrite: bool, 
        batch_rows: int = 20_000, 
        log_every_batches: int = 1,
) -> None:
    if not parquet_path.exists():
        raise FileNotFoundError(f"No parquet found under {parquet_path}")
    total_rows = parquet_num_rows(parquet_path)
    eta = ETAEstimator(total=total_rows)
    log.info(f"Parquet {parquet_path.name}: {total_rows} rows")
    df = normalize_columns(pl.read_parquet(parquet_path))
    if total_rows > batch_rows:
        save_dir.mkdir(parents=True, exist_ok=True)
        for part_id, batch in enumerate(df.iter_slices(n_rows=batch_rows)):
            out_path = save_dir / f"part-{part_id:06d}.parquet"
            if out_path.exists() and not overwrite:
                eta.update(len(batch))
                continue
            results, errors = parallel_extractor(batch)
            report_errors(errors, len(batch))
            results.write_parquet(out_path)
            eta.update(len(batch))
            if (part_id + 1) % log_every_batches == 0:
                log.info(f"part {part_id}, progress {eta.fmt()}")
    else:
        out_path = save_dir.with_suffix(".parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not overwrite:
            log.info(f"{out_path} exists, skipping.")
            return
        log.info(f"Reading {parquet_path}")
        results, errors = parallel_extractor(df)
        report_errors(errors, df.height)
        results.write_parquet(out_path)
        eta.update(df.height)
    log.info(f"Directory done. {eta.fmt()}")

def process_parquets_recursive(
    input_path: Path, 
    parallel_extractor: Callable, 
    save_dir: Path, 
    overwrite: bool, 
) -> None:
    if input_path.is_dir(): # Search for parquets recursively
        parquets = sorted(input_path.rglob("*.parquet"))
        if not parquets:
            raise FileNotFoundError(f"No parquets found under directory {input_path}")
        log.info("Found %d parquet(s).", len(parquets))
        for parquet_path in parquets:
            rel = parquet_path.relative_to(input_path)
            out_dir = save_dir / rel.parent / rel.stem
            process_parquet(parquet_path, parallel_extractor=parallel_extractor, save_dir=out_dir, overwrite=overwrite)
    elif input_path.is_file(): 
        if input_path.suffix != '.parquet':
            raise ValueError(f"Expected a .parquet file, got {input_path}")
        out_dir = save_dir / input_path.stem
        process_parquet(input_path, parallel_extractor=parallel_extractor, save_dir=out_dir, overwrite=overwrite)
    else:
        raise FileNotFoundError(f"Path does not exist: {parquet_path_or_dir}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute time/frequency-domain features for FITS light curves.",
    )
    p.add_argument("mode", choices=("preparation", "training", "inference"), help="Whether to run on the training set or inference set.")
    p.add_argument("input", type=Path, help="In training: Path to a single parquet. In inference: Path to a directory containing of parquets.")
    p.add_argument("output", type=Path, help="Directory to save the extracted features")
    p.add_argument("--config", type=Path, default=None, help="Path to pipeline config JSON. Uses default config if not provided.")
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--backend", choices=("threading", "loky"), default="loky", help="joblib backend")
    p.add_argument("--flux-type", choices=("psf", "aper"), default="aper", help="Flux columns to extract from FITS files.")
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False, help="Overwrite existing output part files.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    cfg = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()
    log.info(f"Config: {cfg}")

    match args.mode:
        case 'preparation':
            extractor = partial(extract_light_curve_from_slice, flux_type=args.flux_type)
            output_dir = args.output / 'light_curves'            
        case 'inference':
            extractor = partial(extract_features_from_slice, cfg=cfg)
            output_dir = args.output / 'features'
        case 'training':
            extractor = partial(extract_features_from_slice, cfg=cfg)
            output_dir = args.output / 'features' / args.input.name
        case _:
            raise ValueError('Invalid mode')
    parallel_extractor = partial(
        extract_in_parallel, 
        extractor_fn=extractor, 
        n_jobs=args.n_jobs, backend=args.backend, worker_batch_size=100
    )
    process_parquets_recursive(args.input, parallel_extractor, output_dir, args.overwrite)

