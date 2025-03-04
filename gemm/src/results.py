import polars

class Results:
    def __init__(self, filename: str):
        self.df = polars.read_json(filename)


