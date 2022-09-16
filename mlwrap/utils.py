import pandas as pd

def to_numpy(X):
    return X.to_numpy() if hasattr(X, "to_numpy") else X


def is_categorical(dtype):
    return pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(
        dtype
    )