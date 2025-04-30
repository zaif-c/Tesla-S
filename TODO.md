# ✅ TODO.md — Tesla-S: Tesla Stock Prediction Project Tracker

This file tracks all key tasks in the pipeline — from setup to deployment. Checkboxes serve as both progress tracking and documentation.

---

## 🔧 Phase 1: Project Setup & Structure

- [x] Initialize Git repo
- [x] Create base folders: `data/`, `notebooks/`, `src/`, `models/`, `tests/`
- [x] Add placeholder files to each folder
- [x] Add `.gitignore`
- [x] Create initial `README.md`
- [x] Add `requirements.txt`
- [x] Add `LICENSE` (MIT preferred)
- [x] Add `TODO.md` for progress tracking
- [x] Finalize pipeline and documentation

---

## 📥 Phase 2: Data Loading & Preprocessing

> Goal: Prepare dataset for analysis and modeling

- [ ] Download Tesla dataset and save to `data/raw/`
- [ ] `src/data_loader.py`:
  - [ ] Load CSV
  - [ ] Convert `Date` to datetime and set as index
  - [ ] Sort chronologically
  - [ ] Handle missing/null values
- [ ] `src/features.py`:
  - [ ] Create moving averages (MA5, MA10, MA20)
  - [ ] Create rolling volatility (e.g., std dev of past 10 days)
  - [ ] Calculate daily and monthly returns
  - [ ] (Optional) Normalize or standardize features
- [ ] Save processed data to `data/processed/`
- [ ] Document steps in `01_preprocessing_eda.ipynb`

---

## 📊 Phase 3: Exploratory Data Analysis (EDA)

> Goal: Understand the data and guide model choices

- [ ] Plot raw price, volume, and returns over time
- [ ] Visualize moving averages and volatility trends
- [ ] Correlation heatmap of features
- [ ] Identify any outliers (e.g., market events, stock splits)
- [ ] Seasonal Decomposition (to decide between ARIMA and SARIMA)
- [ ] Summarize insights in `01_preprocessing_eda.ipynb`

---

## 🤖 Phase 4: Modeling

> Goal: Build, tune, and compare multiple forecasting models

### 🟢 Random Forest
- [ ] `src/model_rf.py`: Build and train `RandomForestRegressor`
- [ ] Tune hyperparameters (n_estimators, max_depth, etc.)
- [ ] Analyze feature importance
- [ ] Save trained model to `models/`
- [ ] Document in `02_model_baselines.ipynb`

### 🟢 XGBoost
- [ ] `src/model_xgb.py`: Build and train `XGBoostRegressor`
- [ ] Tune hyperparameters with GridSearchCV or manual tuning
- [ ] Save trained model to `models/`
- [ ] Document in `02_model_baselines.ipynb`

### 🟡 ARIMA / SARIMA (pick one after EDA)
- [ ] `src/model_arima.py`
- [ ] Run stationarity tests (ADF, KPSS)
- [ ] Tune parameters (p,d,q) / (P,D,Q,s)
- [ ] Forecast closing prices
- [ ] Save model or forecasts
- [ ] Document results in `02_model_baselines.ipynb`

---

## 📈 Phase 5: Evaluation & Analysis

> Goal: Quantify and visualize model performance

- [ ] Define evaluation metrics:
  - [ ] MAE
  - [ ] RMSE
  - [ ] R² Score
- [ ] Implement walk-forward validation
- [ ] Plot actual vs predicted prices for each model
- [ ] Summarize performance in a results table
- [ ] Discuss strengths/weaknesses of each model
- [ ] Document in `04_evaluation.ipynb`

---

## 🔮 Phase 6: (Optional) Deep Learning — LSTM

> Goal: Experiment with sequence modeling

- [ ] `src/model_lstm.py`:
  - [ ] Frame sequences with sliding window (X: past N days, y: next day/month)
  - [ ] Build LSTM using TensorFlow (Keras)
  - [ ] Normalize sequences and split into train/test
  - [ ] Train model and monitor loss
  - [ ] Save model to `models/`
- [ ] Compare LSTM results with ML baselines
- [ ] Document in `03_model_lstm.ipynb`

---

## 🧪 Phase 7: Testing & Utilities

- [ ] `tests/test_features.py`: Unit tests for feature functions
- [ ] `src/utils.py`: Utility functions for evaluation, plotting, etc.
- [ ] Add docstrings to all `src/` functions

---

## 📜 Phase 8: Documentation & Polish

- [ ] Finalize and clean `README.md`
  - [ ] Add final results
  - [ ] Add model comparison table
  - [ ] Add how to run section
- [ ] Add visuals (e.g., prediction plots) to README or repo
- [ ] (Optional) Add GIF, badges, or diagram
- [ ] Clean up notebooks (remove dead cells, clear outputs)
- [ ] Add final commit with message: `Finalize v1`

---

## 🚀 Future Scope (Optional)

- [ ] Integrate macroeconomic data or sentiment (e.g., Twitter, Fed announcements)
- [ ] Explore multivariate time series forecasting
- [ ] Ensemble model combining XGBoost + LSTM
- [ ] Deploy as a simple Streamlit or Flask app
- [ ] Build a real-time data pipeline

---
