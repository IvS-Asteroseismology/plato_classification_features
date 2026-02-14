from typing import Any, Sequence, Callable, Iterator
from joblib import Parallel, delayed


def iter_batches(seq: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]

def extract_in_parallel(
    data: list[dict[str, Any]],
    extractor_fn: Callable,
    *,
    n_jobs: int,
    backend: str,
    worker_batch_size: int,
) -> list[dict[str, Any]]:
    def _extract_many(xs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [extractor_fn(x) for x in xs]

    lists = Parallel(n_jobs=n_jobs, backend=backend, batch_size=1)(
        delayed(_extract_many)(sub) for sub in iter_batches(data, worker_batch_size)
    )
    return [row for sub in lists for row in sub]


