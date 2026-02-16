import re
import pandas as pd


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize feature names by replacing non-alphanumeric characters with underscores.

    Returns:
        pd.DataFrame: DataFrame with sanitized feature names.
    """
    df = df.copy()
    df.columns = [
        re.sub(r"[^A-Za-z0-9_]", "_", col)
        for col in df.columns
    ]
    return df
