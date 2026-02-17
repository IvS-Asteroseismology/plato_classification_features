from collections import Counter
from pathlib import Path
from functools import partial
from typing import Iterator, Any, Callable
import argparse
import logging
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from parallel_utils import extract_in_parallel
from extractors import ExtractorFn, extract_features_from_row, extract_light_curve_from_row
from eta_estimator import ETAEstimator


def iter_parquet_column(
        parquet_path: Path | str, 
        *, 
        batch_rows: int, 
        cols: list[str] | None = None,
) -> Iterator[list[dict[str, Any]]]:
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(columns=cols, batch_size=batch_rows):
        yield batch.to_pylist()

def parquet_num_rows(parquet_path: str | Path) -> int:
    pf = pq.ParquetFile(parquet_path)
    return pf.metadata.num_rows

def write_parquet_part(rows: list[dict[str, Any]], out_path: Path) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)

def prefix_key(p: str, keep_parts: int = 10) -> str:
    # ['', 'STER','mamsie','tess','TIC','plato_fov','s0001','cam3-ccd4','0046','7448', ...]
    parts = p.split('/')
    return '/'.join(parts[:keep_parts])

def remove_and_report_errors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failed = [row for row in rows if '_error' in row]
    if len(failed) > 0:
        log.error(f"{len(failed)}/{len(rows)} files failed")
        errors = [row['_error'] for row in failed]
        cerrors = Counter(errors)
        for reason, count in cerrors.items():
            log.error(f"\t {reason}: {count}")
    return [row for row in rows if '_error' not in row]

def process_parquet(
        parquet_path: Path,
        extractor_fn: ExtractorFn,
        save_dir: Path, 
        n_jobs: int, 
        overwrite: bool, 
        joblib_backend: str, 
        batch_rows: int = 20_000, 
        worker_batch: int = 20, 
        log_every_batches: int = 1,
) -> None:
    if not parquet_path.exists():
        raise FileNotFoundError(f"No parquet found under {parquet_path}")
    total_rows = parquet_num_rows(parquet_path)
    eta = ETAEstimator(total=total_rows)
    log.info(f"Parquet {parquet_path.name}: {total_rows} rows")
    if save_dir.suffix.lower() == ".parquet":
        save_dir.parent.mkdir(parents=True, exist_ok=True)
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
    parallel_extractor = partial(extract_in_parallel, extractor_fn=extractor_fn)
    
    if total_rows > batch_rows:
        for part_id, batch in enumerate(iter_parquet_column(parquet_path, batch_rows=batch_rows)):
            out_path = save_dir / f"part-{part_id:06d}.parquet"
            if out_path.exists() and not overwrite:
                eta.update(len(batch))
                continue
            # batch.sort(key=lambda p: prefix_key(p, keep_parts=10))
            rows = parallel_extractor(batch, n_jobs=n_jobs, backend=joblib_backend, worker_batch_size=worker_batch)
            rows = remove_and_report_errors(rows)
            write_parquet_part(rows, out_path)
            eta.update(len(batch))
            if (part_id + 1) % log_every_batches == 0:
                log.info(f"part {part_id}, progress {eta.fmt()}")
    else:
        if save_dir.exists() and save_dir.is_file() and not overwrite:
            log.info(f"File {save_dir} exists, skipping.")
            return
        log.info(f"Reading {parquet_path}")
        batch = pl.read_parquet(parquet_path).to_dicts()
        rows = parallel_extractor(batch, n_jobs=n_jobs, backend=joblib_backend, worker_batch_size=worker_batch)
        rows = remove_and_report_errors(rows)
        write_parquet_part(rows, save_dir)
        eta.update(len(batch))
    log.info(f"Directory done. {eta.fmt()}")

def process_parquets_recursive(parquet_path_or_dir: Path, extractor_fn: ExtractorFn, save_dir: Path, n_jobs: int, overwrite: bool, joblib_backend: str) -> None:
    if parquet_path_or_dir.is_dir(): # Search for parquets recursively
        parquets = sorted(parquet_path_or_dir.rglob("*.parquet"))
        if not parquets:
            raise FileNotFoundError(f"No parquets found under directory {parquet_path_or_dir}")
        log.info("Found %d parquet(s).", len(parquets))
        for parquet_path in parquets:
            rel_parent = parquet_path.relative_to(parquet_path_or_dir)
            if rel_parent == Path("."):
                out_dir = save_dir / parquet_path.stem
            else:
                out_dir = save_dir / rel_parent
            process_parquet(parquet_path=parquet_path, extractor_fn=extractor_fn, save_dir=out_dir, n_jobs=n_jobs, overwrite=overwrite, joblib_backend=joblib_backend)
    elif parquet_path_or_dir.is_file(): 
        if parquet_path_or_dir.suffix != '.parquet':
            raise ValueError(f"Expected a .parquet file, got {parquet_path_or_dir}")
        out_dir = save_dir / parquet_path_or_dir.stem
        process_parquet(parquet_path=parquet_path_or_dir, extractor_fn=extractor_fn, save_dir=out_dir, n_jobs=n_jobs, overwrite=overwrite, joblib_backend=joblib_backend)
    else:
        raise FileNotFoundError(f"Path does not exist: {parquet_path_or_dir}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute time/frequency-domain features for FITS light curves.",
    )
    p.add_argument("mode", choices=("preparation", "training", "inference"), help="Whether to run on the training set or inference set.")
    p.add_argument("input", type=Path, help="In training: Path to a single parquet. In inference: Path to a directory containing of parquets.")
    p.add_argument("output", type=Path, help="Directory to save the extracted features")
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--backend", choices=("threading", "loky"), default="loky", help="joblib backend")
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False, help="Overwrite existing output part files.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, "INFO"),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    log = logging.getLogger("features")

    def run(fn: Callable, extractor: ExtractorFn, out_dir: Path) -> None:
        fn(
            args.input,
            extractor,
            out_dir,
            n_jobs=args.n_jobs,
            overwrite=args.overwrite,
            joblib_backend=args.backend,
        )
    match args.mode:
        case 'preparation':
            run(process_parquets_recursive, extract_light_curve_from_row, args.output / 'light_curves')
        case 'inference':
            run(process_parquets_recursive, extract_features_from_row, args.output / 'features')
        case 'training':
            run(process_parquets_recursive, extract_features_from_row, args.output / 'features' / args.input.name)
        case '_':
            raise ValueError('Invalid mode')

