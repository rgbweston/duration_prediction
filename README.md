# duration_prediction
Duration Prediction using NHS orthopaedic data

## AutoGluon Modeling

The repository contains two scripts:

- `preprocessing.py` – loads the preprocessed training and test CSV files
  from Google Drive (or a local path), performs boolean-based feature
  selection, and reports dataset information.
- `train_autogluon.py` – trains a lightweight AutoGluon GBM model on the
  selected features, evaluates regression metrics, and computes
  permutation feature importance.

Both scripts follow a structure compatible with Google Colab but can also
run locally.
