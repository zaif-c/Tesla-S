import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path

def load_processed_data(file_path: str = 'data/processed/tesla_processed_data.csv') -> pd.DataFrame:
    """
    Load the processed Tesla stock data.
    
    Args:
        file_path (str): Path to the processed CSV file
        
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    return df

def fit_arima_model(df: pd.DataFrame, column: str = 'Close', p: int = 24, d: int = 1, q: int = 24) -> ARIMA:
    """
    Fit the ARIMA model to the data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column to fit the model
        p (int): AR order
        d (int): Differencing order
        q (int): MA order
        
    Returns:
        ARIMA: Fitted ARIMA model
    """
    model = ARIMA(df[column], order=(p, d, q))
    model_fit = model.fit()
    return model_fit

if __name__ == "__main__":
    try:
        # Load the processed data
        df = load_processed_data()
        print("Data loaded successfully.")
        
        # Split the data into training and testing sets
        train_size = int(len(df) * 0.8)
        train_data = df[:train_size]
        test_data = df[train_size:]
        print("Data split into training and testing sets.")
        
        # Fit the ARIMA model on the training data
        model_fit = fit_arima_model(train_data)
        print("ARIMA model fitted successfully.")
        
        # Print the model summary
        print(model_fit.summary())
    except Exception as e:
        print(f"An error occurred: {e}")