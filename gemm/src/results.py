from typing import Any

import json
import polars as pl


class Results:
    def __init__(self, filename: str, *labels):
        labels = list(labels)
        with open(filename, "r") as file:
            data = json.load(file)
        self.context: dict[str, Any] = data["context"]
        self.df = (
            pl.from_dicts(
                data["benchmarks"],
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
