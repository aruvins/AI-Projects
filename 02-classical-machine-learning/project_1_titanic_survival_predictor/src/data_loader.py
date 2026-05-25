import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    print("\nDataset loaded successfully!")
    print(df.head())
    return df