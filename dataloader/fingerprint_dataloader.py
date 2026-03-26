import os
import pandas as pd
import numpy as np
import google.cloud.storage as storage


def _get_label_column_name(df):
    """Find a label column in the parquet table."""
    candidates = ("label", "labels", "Label", 'LABEL', "Labels", "target", "Target", "y", "Y")
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "Fingerprint parquet must contain a label column. "
        f"Expected one of {candidates}, but got columns: {list(df.columns)}"
    )


def _validate_fingerprint_list(value, row_idx):
    """Accept only list-like parquet values and convert to 1D float32 array."""
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(
            "ECFP4 parquet values must be stored as list-like objects, not strings. "
            f"Row {row_idx} has type {type(value).__name__}."
        )

    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"ECFP4 value at row {row_idx} must be 1D, got shape {arr.shape}.")
    return arr

def get_ecfp4_fingerprints(dataset, dataroot, data_type, bucket_name):
    """
    Fetch ECFP4 fingerprints from a GCP bucket or local directory.

    :param dataset: Name of the dataset (e.g., "bbbp").
    :param dataroot: Local directory to store the dataset.
    :param data_type: Type of data ("raw" or "processed").
    :param bucket_name: Name of the GCP bucket (if using GCP).
    :return: NumPy object array where each row is [fingerprint_array, label].
    """
    parquet_file = os.path.join(dataroot, f"{dataset}_train_ECFP4.parquet")

    if bucket_name:
        # Initialize GCP storage client
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Validate paths
        parquet_blob = bucket.blob(f"{dataset}/{data_type}/{dataset}_train_ECFP4.parquet")
        parquet_blob.download_to_filename(parquet_file)

    ecfp4_df = pd.read_parquet(parquet_file)

    if "ECFP4" not in ecfp4_df.columns:
        raise ValueError("Fingerprint parquet must contain a 'ECFP4' column.")
    label_col = _get_label_column_name(ecfp4_df)

    fp_values = ecfp4_df["ECFP4"].tolist()
    label_values = ecfp4_df[label_col].tolist()

    output = np.empty((len(fp_values), 2), dtype=object)
    for i, (fp, label) in enumerate(zip(fp_values, label_values)):
        output[i, 0] = _validate_fingerprint_list(fp, i)
        output[i, 1] = label

    return output