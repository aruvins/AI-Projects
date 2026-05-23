import pandas as pd

def basic_stats(df: pd.DataFrame):
    """ Computes basic statistics for a DataFrame."""
    return df.describe()

def correlations(df: pd.DataFrame):
    """ Computes the correlation matrix for a DataFrame."""
    return df.corr(
    )

def column_stats(df: pd.DataFrame, column: str):
    """ Computes statistics for a specific column."""
    if column in df.columns:
        return {
            'mean': df[column].mean(),
            'median': df[column].median(),
            'std': df[column].std(),
            'min': df[column].min(),
            'max': df[column].max()
        }
    else:
        print(f"Column '{column}' not found in DataFrame.")
        return None