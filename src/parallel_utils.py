from typing import Any, Callable
import polars as pl
from joblib import Parallel, delayed

def extract_in_parallel(
    df: pl.DataFrame,
    extractor_fn: Callable[[pl.DataFrame], dict[str, Any]],
    *,
    n_jobs: int,
    backend: str,
    worker_batch_size: int,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    #def _extract_many(df_slice) -> list[dict[str, Any]]:
    #    return [extractor_fn(df_slice.slice(i, 1)) for i in range(len(df_slice))]
    def _extract_many(df_slice: pl.DataFrame) -> tuple[pl.DataFrame | None, list[dict[str, Any]]]:
        successes, errors = [], []
        for i in range(len(df_slice)):
            result = extractor_fn(df_slice.slice(i, 1))
            if "_error" in result:
                errors.append(result)
            else:
                successes.append(result)
        success_df = pl.DataFrame(successes)
        return success_df, errors
    slices = [
        df.slice(i, worker_batch_size)
        for i in range(0, len(df), worker_batch_size)
    ]
    results = Parallel(n_jobs=n_jobs, backend=backend, batch_size=1)(
        delayed(_extract_many)(sub) for sub in slices
    )
    dfs = [df for df, _ in results]
    errors = [e for _, errs in results for e in errs]
    return pl.concat(dfs), errors