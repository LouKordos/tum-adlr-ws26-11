from __future__ import annotations

from collections.abc import Callable

from torch.utils.data import DataLoader, Dataset, Subset


def _init_worker(_worker_id: int) -> None:
    from torch.utils.data import get_worker_info

    info = get_worker_info()
    if info is None:
        return
    ds = info.dataset
    while isinstance(ds, Subset):
        ds = ds.dataset
    if hasattr(ds, "reset_worker"):
        ds.reset_worker()


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    collate_fn: Callable | None = None,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = 2,
) -> DataLoader:
    use_workers = num_workers > 0
    if persistent_workers is None:
        persistent_workers = use_workers

    # Avoid yielding 0 batches when debugging on tiny datasets (< batch_size).
    if hasattr(dataset, "__len__") and len(dataset) < batch_size:
        drop_last = False

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        worker_init_fn=_init_worker if use_workers else None,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if use_workers else None,
    )