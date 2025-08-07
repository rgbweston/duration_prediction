"""Data preprocessing utilities for duration prediction.

This module handles data loading and feature selection with a boolean
mask to enable or disable specific columns. It is designed to mirror
Google Colab workflows, including optional Drive mounting and runtime
tracking.
"""

# Set paths and load df

# ── Block 1: Mount Drive, imports, and runtime setup ──

use_full_train_test = True
use_sample_train_test = False
filename = 'duration_prediction_6.0'

from pathlib import Path
import json
import os
import pandas as pd

try:  # Optional Colab imports
    from google.colab import drive, files  # type: ignore
    _IN_COLAB = True
except Exception:  # pragma: no cover - executed outside Colab
    _IN_COLAB = False

# Runtime control parameters
runtime_file_name = 'ag_runtime.json'
total_time_cap = 3 * 3600      # total cap in seconds (3 h)
chunk_time_limit = 60 * 60     # optional: max time per chunk
status_interval = 60 * 60      # optional: how often to report status


def prepare_datasets(
    *,
    use_full_train_test: bool = use_full_train_test,
    use_sample_train_test: bool = use_sample_train_test,
    base_dir: str = '/content/drive/MyDrive/UCL_Dissertation',
    filename: str = filename,
):
    """Load training and test data and select columns.

    Parameters
    ----------
    use_full_train_test: bool
        Whether to use the full training and test sets.
    use_sample_train_test: bool
        Whether to use sampled training and test sets.
    base_dir: str
        Base directory where processed CSVs are stored.
    filename: str
        Name used for saving models and runtime files.

    Returns
    -------
    dict
        Contains ``train_data``, ``test_data``, ``model_save_path``,
        ``runtime_file`` and ``elapsed_time``.
    """

    # Mount Drive if running in Colab
    if _IN_COLAB:
        drive.mount('/content/drive', force_remount=True)  # pragma: no cover

    base_path = Path(base_dir)

    # Define paths to processed data
    if use_full_train_test:
        train_data_path = base_path / 'train_data.csv'
        test_data_path = base_path / 'test_data.csv'
    elif use_sample_train_test:
        train_data_path = base_path / 'train_data_sample.csv'
        test_data_path = base_path / 'test_data_sample.csv'
    else:
        raise ValueError("Either use_full_train_test or use_sample_train_test must be True")

    # Load train data (fallback to upload if file not found)
    if not train_data_path.exists():
        if _IN_COLAB:
            print(f"Uploading {train_data_path.name}…")
            uploaded = files.upload()  # pragma: no cover
            train_data_path = Path(list(uploaded.keys())[0])
        else:
            raise FileNotFoundError(train_data_path)
    train_data = pd.read_csv(train_data_path)

    # Load test data (fallback to upload if file not found)
    if not test_data_path.exists():
        if _IN_COLAB:
            print(f"Uploading {test_data_path.name}…")
            uploaded = files.upload()  # pragma: no cover
            test_data_path = Path(list(uploaded.keys())[0])
        else:
            raise FileNotFoundError(test_data_path)
    test_data = pd.read_csv(test_data_path)

    # ── Remove 'Expected_Length' if present ──
    for df_name, df in [('train_data', train_data), ('test_data', test_data)]:
        if 'Expected_Length' in df.columns:
            df.drop(columns=['Expected_Length'], inplace=True)
            print(f"▶ Removed 'Expected_Length' from {df_name}")

    # Paths for saving model
    model_save_path = base_path / 'MyModels' / filename
    model_save_path.mkdir(parents=True, exist_ok=True)

    # Runtime file
    runtime_file = model_save_path / runtime_file_name

    # Load or initialize elapsed time
    if runtime_file.exists():
        with open(runtime_file, 'r') as f:
            elapsed_time = json.load(f).get('elapsed_time', 0.0)
    else:
        elapsed_time = 0.0

    # Print runtime and data info
    print(f"▶ Already used {elapsed_time/60:.1f} min of {total_time_cap/3600:.1f} h cap.")
    print(f"▶ Loaded train data from: {train_data_path} (shape {train_data.shape})")
    print(f"▶ Loaded test  data from: {test_data_path} (shape {test_data.shape})")

    # Print first line of training data
    print(train_data.head(1))
    print(train_data.dtypes)

    # Feature inclusion map (boolean column selection)
    include_cols = {
        'Year': True,
        'Operation_Age': True,
        'Sex': True,
        'Ethnic_Category': True,
        'Anaesthetic_Type_Code': True,
        'Theatre_Code': True,
        'Proc_Code_1_Read': True,
        'Intended_Management': True,
        'IMD_Score': True,
        'High_Volume_and_Low_Complexity_Category': True,
        'High_Volume_and_Low_Complex_or_not': True,
        'weekday_name': True,
        'day_working': True,
        'Season': True,
        'Pseudo_Consultant': True,
        'Pseudo_Surgeon': True,
        'Pseudo_Anaesthetist': True,
        'Previous_Operation_Length': True,
        'Previous_Proc_1': True,
        'Heart_Condition': True,
        'Hypertension': True,
        'Obesity': True,
        'Diabetes': True,
        'Cancer': True,
        'Chronic_Kidney_Disease': True,
        'Surgeon_Anaesthetist_Team': True,
        'Proc_Code_1_Read_pc1': True,
        'Previous_Proc_1_pc1': True,
        'Previous_Operation': True,
        'is_rare_anaes': True,
        'is_rare_Proc_Code_1_Read': True,
        'Proc_Code_1_Read_te_smooth': True,
        'Anaesthetic_te_smooth': True,
    }

    cols_to_use = [col for col, use in include_cols.items() if use]
    cols_to_use.append('Actual_Length')

    train_data = train_data[cols_to_use].copy()
    test_data = test_data[cols_to_use].copy()

    print(f"▶ Using {len(cols_to_use)-1} features + target:")
    print(cols_to_use[:-1])

    return {
        'train_data': train_data,
        'test_data': test_data,
        'model_save_path': model_save_path,
        'runtime_file': runtime_file,
        'elapsed_time': elapsed_time,
    }


__all__ = ["prepare_datasets"]
