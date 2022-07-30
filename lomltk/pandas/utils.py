from typing import Iterable, Optional

from pandas import DataFrame

from ..utils import get_progress_bar

__all__ = [
    "merge_dataframes",
]


def merge_dataframes(dataframes: Iterable[DataFrame]) -> Optional[DataFrame]:
    output_df: Optional[DataFrame] = None

    for df in get_progress_bar(dataframes, desc="Merging"):
        if output_df is None:
            output_df = df
        else:
            output_df = output_df.merge(
                df,
                how="outer",
                on=output_df.columns.intersection(df.columns).tolist(),
                validate="1:1"
            )

    return output_df
