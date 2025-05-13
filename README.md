# Tesla-S 
## Project Overview
Financial time series forecasting is a key challenge in quantitative finance. Tesla, being one of the most volatile and actively traded stocks, serves as an ideal candidate to study and forecast price trends. This project aims to develop supervised machine learning models and deep learning-based architectures to forecast the next month's closing price of Tesla stock using historical monthly data (Date, Open, High, Low, Close, Volume) and to establish strong baseline and advanced benchmarks.
The project focuses purely on the machine learning pipeline: data preprocessing, exploratory data analysis, feature engineering, model training, and evaluation of different forecasting techniques, comparing their results in a meaningful and interpretable manner.

## Problem Statement
Given historical day stock data for Tesla, predict the closing price for the following month at least. Evaluate which methods perform best under different market conditions and quantify prediction accuracy using appropriate error metrics. Perform necessary EDA and comparative analysis.

## Dataset
**Source:** [Kaggle - Tesla Historical Stock Prices](https://www.kaggle.com/datasets/jillanisofttech/tesla-stock-price)\
\
**Dataset Features:**
- Date
- Open, High, Low, Close
- Volume

## Tech Stack
**Data Handling & Processing**: `pandas`, `numpy`  
**Visualization**: `matplotlib`, `seaborn`, `plotly`  
**Modeling**:
- Traditional: `ARIMA`, `SARIMA` <-- CHOOSE ONE BASED ON EDA !!!
- Machine Learning: `RandomForestRegressor`, `XGBoost`
 

**Evaluation**: `scikit-learn` metrics (MAE, MSE, RMSE, R²)
## Approach/Pipeline

### 1️⃣ Data Collection & Preprocessing

> **Objective**: Prepare the dataset for analysis and modeling.

- Load the Tesla historical stock dataset (CSV) using `pandas`.
- Convert `Date` column to datetime format and set as index.
- Sort chronologically and check for missing/null values.
- Handle missing data (impute or drop as appropriate).
- Create time-aware features:
  - Moving averages (MA5, MA10, MA20)
  - Rolling volatility (e.g., 10-day std deviation)
  - Daily and monthly returns
- Normalize or standardize features if needed.
- Save the cleaned dataset in `data/processed/`.

📁 Notebook: `01_preprocessing_eda.ipynb`  
📁 Code: `src/data_loader.py`, `src/features.py`

---

### 2️⃣ Exploratory Data Analysis (EDA)

> **Objective**: Understand trends, relationships, and seasonality in the data.

- Plot `Close` price, volume, returns, and moving averages over time.
- Check for seasonality or patterns to inform ARIMA vs. SARIMA decision.
- Visualize:
  - Volatility clustering
  - Price vs. volume movement
  - Correlation heatmap of features
- Detect and document outliers (e.g., splits, Elon tweets, market crashes).

📁 Notebook: `01_preprocessing_eda.ipynb`

---

### 3️⃣ Baseline and Traditional Modeling

> **Objective**: Train simple yet interpretable models as performance baselines.

- **Random Forest Regressor** (scikit-learn)
  - Use structured features like MA, returns, and volume
  - Analyze feature importances
- **XGBoost Regressor**
  - Apply same feature set; compare against RF
  - Fine-tune using grid search or cross-validation
- **ARIMA or SARIMA** (choose based on EDA):
  - Model closing price or returns directly
  - Perform stationarity tests (ADF)
  - Tune `(p,d,q)` or `(P,D,Q,s)` using AIC/BIC

📁 Notebook: `02_model_baselines.ipynb`  
📁 Code: `src/model_rf.py`, `src/model_xgb.py`, `src/model_arima.py`

---

### 4️⃣ Model Evaluation

> **Objective**: Compare models using quantitative metrics and visual insights.

- Evaluation Metrics:
  - **MAE** (Mean Absolute Error)
  - **RMSE** (Root Mean Squared Error)
  - **R² Score** (Explained Variance)
- Use **walk-forward validation** to simulate realistic forecasting
- Plot actual vs. predicted closing prices
- Highlight performance under volatile periods (e.g., 2020 COVID crash)

📁 Notebook: `04_evaluation.ipynb`  
📁 Code: `src/train.py`, `src/utils.py`

---

### 5️⃣ (Optional) Deep Learning Extension – LSTM

> **Objective**: Experiment with sequence-based modeling if time allows.

- Restructure time series into sliding windows (e.g., 30 days → 1 target)
- Normalize input sequences
- Define and train LSTM using `TensorFlow` or `PyTorch`
- Evaluate using the same walk-forward method and compare to ML baselines

📁 Notebook: `03_model_lstm.ipynb`  
📁 Code: `src/model_lstm.py`

---

### 6️⃣ Results and Analysis

> **Objective**: Summarize insights, evaluate models, and draw conclusions.

- Present metrics and charts in a summary table
- Compare performance under stable vs. volatile conditions
- Discuss tradeoffs: interpretability vs. accuracy vs. complexity
- Identify best-performing model and why

📁 Notebook: `04_evaluation.ipynb`

---

### 7️⃣ Future Work

> Extend the project with more advanced data sources and modeling techniques.

- Deep Learning: `LSTM` (via `TensorFlow` or `PyTorch`)
- Use ensemble methods combining RF, XGBoost, and LSTM
- Deploy a real-time prediction pipeline using streaming APIs
## Results
## How to Run
