from collections import Counter
from pathlib import Path
from typing import Iterator, Any
import argparse
import logging
import pyarrow as pa
import pyarrow.parquet as pq

from feature_extractors import extract_features_parallel
from eta_estimator import ETAEstimator

def iter_parquet_column(parquet_path: Path | str, *, batch_rows: int, col: str) -> Iterator[list[str]]:
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(columns=[col], batch_size=batch_rows):
        yield batch.column(0).to_pylist()

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

def process_parquet(
        parquet_path: Path,
        save_dir: Path, 
        n_jobs: int, 
        overwrite: bool, 
        joblib_backend: str, 
        batch_rows: int = 20_000, 
        worker_batch: int = 20, 
        log_every_batches: int = 1,
        verbose: bool = False,
        path_column_name: str = 'FilePath',
) -> None:
    if not parquet_path.exists():
        raise FileNotFoundError(f"No parquet found under {parquet_path}")
    total_rows = parquet_num_rows(parquet_path)
    eta = ETAEstimator(total=total_rows)
    log.info(f"Parquet {parquet_path.stem}: {total_rows} paths")
    save_dir.mkdir(parents=True, exist_ok=True)

    for part_id, batch in enumerate(iter_parquet_column(parquet_path, batch_rows=batch_rows, col=path_column_name)):
        out_path = save_dir / f"part-{part_id:06d}.parquet"
        if out_path.exists() and not overwrite:
            eta.update(len(batch))
            continue
        batch.sort(key=lambda p: prefix_key(p, keep_parts=10))
        rows = extract_features_parallel(
            batch,
            n_jobs=n_jobs,
            backend=joblib_backend,
            worker_batch_size=worker_batch,
        )
        failed = [row for row in rows if '_error' in row]
        if verbose:
            for row in failed:
                log.error(f"Failed: {row['path']} | {row['_error']}")
        if len(failed) > 0:
            log.error(f"Batch {part_id}: {len(failed)}/{len(rows)} files failed")
            errors = [row['_error'] for row in failed]
            cerrors = Counter(errors)
            for reason, count in cerrors.items():
                log.error(f"\t {reason}: {count}")
        rows = [row for row in rows if '_error' not in row]
        write_parquet_part(rows, out_path)
        eta.update(len(batch))
        if (part_id + 1) % log_every_batches == 0:
            log.info(f"part {part_id}, progress {eta.fmt()}")
    log.info(f"Directory done. {eta.fmt()}")

def process_many_parquets(parquet_dir: Path, save_dir: Path, n_jobs: int, overwrite: bool, joblib_backend: str) -> None:
    parquets = sorted(parquet_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquets found under {parquet_dir}")
    log.info("Found %d parquet(s).", len(parquets))
    for parquet_path in parquets:
        process_parquet(
            parquet_path=parquet_path, 
            save_dir=save_dir / parquet_path.stem, 
            n_jobs=n_jobs, 
            overwrite=overwrite, 
            joblib_backend=joblib_backend,
        )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute time/frequency-domain features for FITS light curves.",
    )
    p.add_argument("mode", choices=("training", "inference"), help="Whether to run on the training set or inference set.")
    p.add_argument("input", type=Path, help="In training: Path to a single parquet. In inference: Path to a directory containing of parquets.")
    p.add_argument("output", type=Path, help="Directory to save the extracted features")
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--backend", choices=("threading", "loky"), default="threading", help="joblib backend")
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False, help="Overwrite existing output part files.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, "INFO"),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    log = logging.getLogger("features")
    match args.mode:
        case 'inference':
            process_many_parquets(args.input, args.output / 'features', args.n_jobs, args.overwrite, args.backend)
        case 'training':
            process_parquet(args.input, args.output / 'features' / 'training_set', args.n_jobs, args.overwrite, args.backend, path_column_name='path')
        case '_':
            raise ValueError('Invalid mode')

