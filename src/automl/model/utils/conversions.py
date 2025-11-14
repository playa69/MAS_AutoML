import numpy as np
import pandas as pd


def convert_to_numpy(data):
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        # Handle categorical/string data for classification tasks
        if isinstance(data, pd.Series):
            if pd.api.types.is_categorical_dtype(data):
                # Convert categorical to integer codes
                return data.cat.codes.to_numpy(dtype=np.int32)
            elif pd.api.types.is_object_dtype(data):
                # Convert string labels to integer codes using factorize
                codes, _ = pd.factorize(data)
                return codes.astype(np.int32)
        
        # Try to convert to float32, fall back to int if that fails
        try:
            return data.to_numpy(dtype=np.float32)
        except (ValueError, TypeError):
            # If conversion to float fails, try integer (for classification labels)
            try:
                if isinstance(data, pd.Series):
                    codes, _ = pd.factorize(data)
                    return codes.astype(np.int32)
                return data.to_numpy(dtype=np.int32)
            except (ValueError, TypeError):
                # Last resort: convert without specifying dtype
                return data.to_numpy()

    if isinstance(data, np.ndarray):
        return data

    if isinstance(data, list):
        # Try float32 first, fall back to int if needed
        try:
            return np.array(data, dtype=np.float32)
        except (ValueError, TypeError):
            return np.array(data, dtype=np.int32)

    raise AttributeError(
        "Input data is of incorrect type. Supported types: 'pandas' ,'numpy'"
    )


def convert_to_pandas(data):
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data

    if isinstance(data, np.ndarray):
        return pd.DataFrame(data, columns=[f"column_{i}" for i in range(data.shape[1])])

    raise AttributeError(
        "Input data is of incorrect type. Supported types: 'pandas' ,'numpy'"
    )
