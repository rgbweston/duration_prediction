"""Training script using AutoGluon with boolean feature selection.

This script relies on :mod:`preprocessing` for data loading and column
selection. It trains a lightweight Gradient Boosting Machine (GBM) model
and reports common regression metrics along with permutation feature
importance.
"""

# Set paths and load df

# ── Block 1: Mount Drive, imports, and runtime setup ──

import json
import time

from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import prepare_datasets

# Obtain preprocessed data and paths
context = prepare_datasets()
train_data = context['train_data']
test_data = context['test_data']
model_save_path = context['model_save_path']
runtime_file = context['runtime_file']
elapsed_time = context['elapsed_time']

# Reload data into AutoGluon datasets
train_data = TabularDataset(train_data)
test_data = TabularDataset(test_data)

# Force clean model folder
if model_save_path.exists():
    import shutil
    shutil.rmtree(model_save_path)
    print("▶ Old model directory removed.")

print("▶ Training lightweight model (GBM only, time_limit=60s)...")
start_time = time.time()

predictor = TabularPredictor(
    label='Actual_Length',
    problem_type='regression',
    eval_metric='mean_absolute_error',
    path=str(model_save_path),
    verbosity=2,
).fit(
    train_data=train_data,
    time_limit=60,
    presets='medium_quality',
    hyperparameters={'GBM': {}},
    num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
)

train_duration = time.time() - start_time
print(f"▶ Training time: {train_duration/60:.1f} minutes")

# Update runtime file
new_elapsed = elapsed_time + train_duration
with open(runtime_file, 'w') as f:
    json.dump({'elapsed_time': new_elapsed}, f)

# ── METRICS ──
print("▶ Evaluating on test set...")
y_true = test_data['Actual_Length']
y_pred = predictor.predict(test_data)

mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)

print(f"📊 Test MAE:   {mae:.2f}")
print(f"📊 Test RMSE:  {rmse:.2f}")
print(f"📊 Test R²:    {r2:.3f}")

# ── Permutation Importance ──
print("▶ Calculating permutation feature importance (30% of test set)...")
subsample_n = int(0.3 * len(test_data))
fi = predictor.feature_importance(data=test_data, subsample_size=subsample_n)

fi_sorted = fi[['importance']].sort_values(by='importance', ascending=False)
print(fi_sorted)
