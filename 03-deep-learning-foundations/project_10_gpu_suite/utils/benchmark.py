from pathlib import Path
import pandas as pd


class BenchmarkLogger:
    def __init__(self, save_dir="./outputs/logs",):
        self.save_dir = Path(save_dir)

        self.save_dir.mkdir(parents=True,exist_ok=True,)
        self.results = []

    def log(self, result):
        self.results.append(result)

    def to_dataframe(self):
        return pd.DataFrame(self.results)

    def save_csv(self,filename="benchmark_results.csv",):
        df = self.to_dataframe()
        path = self.save_dir / filename
        df.to_csv(path, index=False,)
        print(f"Saved benchmark to {path}")
        return path

    def summary(self):
        df = self.to_dataframe()
        print("\n===== Benchmark Summary =====\n")
        print(df)
        return df