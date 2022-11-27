from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .distributed import is_distributed

__all__ = [
    "create_dataloader",
]


def create_dataloader(dataset: Dataset, **kwargs) -> DataLoader:
    sampler = kwargs.pop("sampler", None)
    batch_sampler = kwargs.pop("batch_sampler", None)

    if sampler is not None:
        # Mutually exclusive with `shuffle`.
        kwargs.pop("shuffle", None)

    if batch_sampler is not None:
        # Mutually exclusive with `batch_size`, `shuffle`, `sampler`, and `drop_last`.
        kwargs.pop("batch_size", None)
        kwargs.pop("shuffle", None)
        kwargs.pop("drop_last", None)

    if is_distributed() and sampler is None and batch_sampler is None:
        # Mutually exclusive with `shuffle`, and `drop_last`.
        sampler = DistributedSampler(
            dataset,
            shuffle=kwargs.pop("shuffle", True),
            drop_last=kwargs.pop("drop_last", False)
        )

    return DataLoader(
        dataset,
        sampler=sampler,
        batch_sampler=batch_sampler,
        **kwargs
    )
