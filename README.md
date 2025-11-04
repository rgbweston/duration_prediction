# Surgery Duration Prediction with AutoML

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automated machine learning framework for predicting surgical case duration in NHS operating theatres, achieving **46% improvement** over traditional surgeon estimates.

## 📊 Overview

This repository contains the implementation of a surgery duration prediction system using AutoGluon, an automated machine learning (AutoML) framework. The study evaluated 94,502 elective orthopaedic procedures from East Kent Hospitals University NHS Foundation Trust (2010-2025), comparing AutoGluon against traditional machine learning baselines.

### Key Results

| Model | MAE (minutes) | R² | Improvement |
|-------|---------------|-----|-------------|
| **AutoGluon (1 hour)** | **15.70** | **0.77** | 26% vs XGBoost |
| **AutoGluon (4 hours)** | **11.84** | **0.88** | 46% vs surgeon estimates |
| XGBoost | 16.02 | 0.77 | — |
| Neural Network | 17.67 | — | — |
| Linear Regression | 20.25 | 0.69 | — |
| Surgeon Estimates (baseline) | ~59-70 | — | — |

## 🎯 Motivation

**Operating room inefficiencies cost the NHS over £400 million annually.** Traditional duration estimation methods are inaccurate:
- Surgeon estimates: 59-70 minute mean absolute error (MAE)
- Historical averages: 31-38 minute MAE

With 7.4 million patients on NHS waiting lists (March 2025), accurate prediction is critical for:
- Reducing theatre overruns and staff overtime
- Minimizing idle time between cases
- Maximizing theatre utilisation
- Enabling data-driven scheduling decisions

## 🏗️ Project Structure

```
duration_prediction/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
│
├── Data Preprocessing
│   └── preprocessing.py         # Data loading and feature selection
│
├── Model Training
│   ├── train_autogluon.py       # AutoGluon training (lightweight)
│   ├── train_baseline_models.py # LR, XGBoost, NN comparison
│   └── train_autogluon_full.py  # Extended AutoGluon (4-hour ceiling)
│
├── Feature Analysis
│   ├── shap_duration.py         # SHAP for duration prediction
│   ├── shap_overrun.py          # SHAP for overrun classification
│   └── feature_importance.py    # AutoGluon permutation importance
│
└── Results & Visualizations
    ├── model_comparison.py      # Performance metrics visualization
    └── shap_visualizations.py   # SHAP plots and analysis
```

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip or conda
```

### Installation

```bash
# Clone the repository
git clone https://github.com/rgbweston/duration_prediction.git
cd duration_prediction

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Data Preprocessing

```bash
python preprocessing.py
```

This loads training/test data and performs boolean-based feature selection. Compatible with Google Colab (Drive mounting) or local execution.

#### 2. Train AutoGluon Model

**Lightweight (1 hour):**
```bash
python train_autogluon.py
```

**Extended (4 hours, ceiling performance):**
```bash
python train_autogluon_full.py
```

#### 3. Train Baseline Models for Comparison

```bash
python train_baseline_models.py
```

Trains and evaluates:
- Linear Regression
- XGBoost (with Optuna hyperparameter optimization)
- Feed-Forward Neural Network

#### 4. SHAP Feature Analysis

```bash
# Duration prediction model
python shap_duration.py

# Overrun classification model
python shap_overrun.py
```

#### 5. Visualize Results

```bash
python model_comparison.py
python shap_visualizations.py
```

## 📈 Methodology

### Dataset

- **Source:** East Kent Hospitals University NHS Foundation Trust
- **Procedures:** 94,502 elective orthopaedic cases (2010-2025)
- **Features:** 36 variables including:
  - Patient demographics (age, sex, ethnicity)
  - Clinical factors (comorbidities, IMD score)
  - Procedure details (code, anaesthetic type, intended management)
  - Scheduling context (year, season, weekday, theatre location)
  - Historical data (previous procedure length)
  - Staff identifiers (surgeon, consultant, anaesthetist)

### Data Preprocessing

**Cleaning:**
- Removed features with >50% missingness
- Handled procedure length outliers
- Imputed missing procedure codes (16.56% of cases) using domain knowledge

**Encoding:**
- **Out-of-fold target encoding** for high-cardinality features (prevents data leakage)
- **One-hot encoding** for low-cardinality features (<20 unique values)

**Split:** 64% training / 16% validation / 20% test

### Model Comparison

All models received **identical preprocessing** and **equal computational budgets**:

1. **Linear Regression** - Simple baseline
2. **XGBoost** - State-of-the-art gradient boosting (Optuna HPO)
3. **Neural Network** - Feed-forward architecture (Optuna HPO)
4. **AutoGluon** - Automated ML with multi-layer stacking

**AutoGluon Variants Tested:**
- `AutoGluon-raw`: Minimal preprocessing (MAE: 15.38)
- `AutoGluon-clean`: Basic cleaning (MAE: 15.40)
- `AutoGluon-processed`: Full preprocessing (MAE: 15.70)
- `AutoGluon-full`: Extended 4-hour training (MAE: 11.84, R²: 0.88)

**Key Finding:** AutoGluon demonstrated robust performance regardless of preprocessing quality, with minimal variation across variants. This suggests **computational resources drive performance gains more than preprocessing refinements**.

### Feature Analysis

**SHAP (SHapley Additive exPlanations)** analysis identified key drivers:

#### Duration Predictors
1. **Intended Management** (inpatient vs day case) - 17.8 min mean SHAP
2. **Procedure Code** - 15.8 min mean SHAP (widest variance: +100 to -50 min)
3. **Anaesthetic Type** - Moderate effect

#### Overrun Risk Drivers
1. **Procedure Code** - 9.9 percentage points mean SHAP (dominant)
2. **Year** - Temporal scheduling changes
3. **Theatre Location** - Equipment and setup differences

**AutoGluon's permutation feature importance corroborated SHAP findings**, strengthening confidence that procedural complexity and organizational allocation are the primary sources of scheduling uncertainty.

## 🔍 Key Insights

### 1. Differential Effect of Intended Management

Intended management emerged as the **strongest predictor of duration** but had **minimal impact on overrun likelihood**. This suggests:
- Schedulers already effectively account for inpatient status when allocating theatre time
- Struggle to accommodate procedure-specific duration variability

**Implication:** Buffer times should be adjusted by **procedure code** rather than uniformly by inpatient status.

### 2. Procedure-Specific Variability

Procedure code showed:
- High duration impact (15.8 min mean SHAP, range: +100 to -50 min)
- Highest overrun risk impact (9.9 percentage points)

**Recommendation:** Batch similar procedure types together to reduce schedule uncertainty.

### 3. Anaesthetic Type

Third most important feature across both models.

**Actionable:** Consider anaesthetic type as a modifiable factor in scheduling optimization.

### 4. Notably Absent Features

- **Surgeon experience** (dataset lacked chronological data for experience trajectory construction)
- **Patient comorbidities** (ASA, age, BMI) - featured prominently in other studies but not here

This discrepancy likely reflects data limitations rather than true clinical irrelevance.

## 📊 AutoGluon Architecture

AutoGluon uniquely incorporates **multi-layer stacking with multi-fold bagging**:

```
Input Data
    ↓
Base Models (Layer 1)
  - LightGBM, CatBoost, XGBoost, RF, etc.
  - Multi-fold bagging for each model
    ↓
Stack Layer 2
  - Ensemble models trained on Layer 1 predictions
  - Out-of-fold predictions maintain independence
    ↓
Stack Layer 3
  - Meta-learner combining all previous layers
    ↓
Final Prediction
```

This maintains independence between layers and enables AutoGluon to safely stack 3+ layers for incremental predictive power.

## 🛠️ Technical Stack

- **AutoML Framework:** AutoGluon
- **ML Libraries:** scikit-learn, XGBoost, PyTorch
- **Hyperparameter Optimization:** Optuna
- **Feature Analysis:** SHAP (TreeExplainer)
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn

## 📚 Citation

If you use this code or methodology, please cite:

```bibtex
@misc{barrowcliff2025surgery,
  author = {Barrowcliff, Rohan and Lovegrove, Thomas and Kunz, Holger},
  title = {Surgery Duration Prediction with AutoML and Feature-Based Scheduling Optimisation},
  year = {2025},
  institution = {University College London, Institute of Health Informatics},
  howpublished = {\url{https://github.com/rgbweston/duration_prediction}}
}
```

## 🔮 Future Work

- **Uncertainty quantification:** Provide confidence intervals for duration predictions to enable risk-informed scheduling decisions
- **Multi-specialty expansion:** Evaluate performance across surgical specialties
- **Real-time deployment:** Integrate into theatre management systems for prospective validation
- **Temporal drift monitoring:** Track feature importance changes over time to detect operational shifts
- **Surgeon experience modeling:** Incorporate granular chronological data to construct experience trajectories

## ⚠️ Limitations

- Limited to elective orthopaedic procedures at a single NHS Trust
- Dataset lacked granular chronological data and staff experience variables
- Prospective validation in live theatre scheduling required to evaluate real-world impact

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

**Rohan Barrowcliff**
- Email: rgbweston@gmail.com
- LinkedIn: [linkedin.com/in/rohanbarrowcliff](https://linkedin.com/in/rohanbarrowcliff)
- Website: [rgbweston.github.io](https://rgbweston.github.io)

## 🙏 Acknowledgments

- East Kent Hospitals University NHS Foundation Trust for providing the dataset
- University College London Institute of Health Informatics
- NHS England for operational context and cost data

---

⭐ If you find this project useful, please consider giving it a star on GitHub!
