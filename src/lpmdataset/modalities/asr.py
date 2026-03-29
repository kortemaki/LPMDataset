from collections.abc import Iterable
from functools import cached_property

import pandas as pd
from pydantic import computed_field, ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ASR:
    path: str

    @computed_field
    @cached_property
    def df(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

    @computed_field
    def tokens(self) -> Iterable[str]:
        return self.df['Word'].tolist()

    def to_string(self) -> str:
        return " ".join(self.tokens)

    @computed_field
    def sentences_path(self) -> str:
        return f"{self.path[:-10]}asrsegments.csv"

    def to_sentences(self) -> list[tuple[str, tuple[int, int]]]:
        return pd.read_csv(self.sentences_path).apply(
            lambda row: (row['Sentence'], (row['Start'], row['End'])),
            axis=1,
        ).tolist()
