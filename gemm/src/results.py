from __future__ import annotations

from typing import Iterable, Any

import json
from math import sqrt
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

class Results:
    context: dict[str, Any]
    df: pl.DataFrame
    labels: tuple[str, ...]

    def __init__(self, filename: str, *args):
        self.labels = tuple(args)
        with open(filename, "r") as file:
            data = json.load(file)
        self.context = data["context"]
        self.df = self._build_df(data=data["benchmarks"], labels=self.labels)

    @staticmethod
    def _build_df(*, data: list[dict[str, Any]], labels: tuple[str, ...]) -> pl.DataFrame:
        return (
            pl.from_dicts(
                data,
                schema={
                    "name": pl.String,
                    "cpu_time": pl.Float64,
                },
            )
            .with_columns(
                pl.col("name")
                .str.split_exact("/", 2 + len(labels))
                .struct.rename_fields(["fixture", "benchmark", *labels])
                .alias("fields")
            )
            .unnest("fields")
            .cast({col: pl.Int64 for col in labels})
            .drop("name")
        )


    def filter(self, *benchmarks: tuple[str]) -> Results:
        result = Results.__new__(Results)
        result.context = self.context
        result.df = self.df.filter(pl.col("benchmark").is_in(list(benchmarks)))
        result.labels = self.labels
        return result
        

    def plot(self, *, x: str, ref_exponent: int | None = None):
        xticks: list[int] = self.df[x].unique().to_list()

        _, ax = plt.subplots(figsize=(8,4))
        ax.set(xscale="log", yscale="log", xticks=xticks, xticklabels=xticks)
        ax.minorticks_off()
        ax.grid()
        sns.lineplot(
            ax=ax, data=self.df, x=x, y="cpu_time", hue="benchmark", marker='o'
        )

        if ref_exponent is not None:
            x_min: float = sqrt(max(xticks) * min(xticks))
            x_max: float = max(xticks)
            y_min: float = min(self.df["cpu_time"].to_list())
            y_max: float = y_min * (float(x_max) / float(x_min))**ref_exponent

            ax.plot(
                [x_min, x_max, x_max, x_min],
                [y_min, y_max, y_min, y_min],
                label=f"O({x}^{ref_exponent})",
                linestyle="--",
            )
        
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)
        return ax
