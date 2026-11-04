from collections.abc import Callable

from torch.utils.data import DataLoader, Dataset


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    collate_fn: Callable | None = None,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
