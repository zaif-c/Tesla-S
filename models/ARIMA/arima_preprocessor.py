import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path

# Create arima_plots directory if it doesn't exist
output_dir = Path('outputs/arima_plots')
output_dir.mkdir(parents=True, exist_ok=True)

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

def check_stationarity(df: pd.DataFrame, column: str = 'Close') -> bool:
    """
    Check if the time series is stationary using the Augmented Dickey-Fuller test.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column to check for stationarity
        
    Returns:
        bool: True if stationary, False otherwise
    """
    result = adfuller(df[column].dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    return result[1] < 0.05

def apply_differencing(df: pd.DataFrame, column: str, order: int = 1) -> pd.DataFrame:
    """
    Apply differencing to the specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column to apply differencing
        order (int): Order of differencing
        
    Returns:
        pd.DataFrame: DataFrame with differenced column
    """
    df_diff = df.copy()
    df_diff[f'{column}_diff'] = df[column].diff(order)
    return df_diff

def plot_acf_pacf(df: pd.DataFrame, column: str, title: str) -> None:
    """
    Plot ACF and PACF for the specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column to plot
        title (str): Title for the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(df[column].dropna(), ax=axes[0])
    plot_pacf(df[column].dropna(), ax=axes[1])
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_dir / f'acf_pacf_{title.lower().replace(" ", "_")}.png')
    plt.close()

if __name__ == "__main__":
    # Load the processed data
    df = load_processed_data()
    
    # Check stationarity
    is_stationary = check_stationarity(df, 'Close')
    print(f"Is the data stationary? {is_stationary}")
    
    # Apply differencing if not stationary
    if not is_stationary:
        df_diff = apply_differencing(df, 'Close')
        is_stationary_diff = check_stationarity(df_diff, 'Close_diff')
        print(f"Is the differenced data stationary? {is_stationary_diff}")
        
        # Plot ACF and PACF for differenced data
        plot_acf_pacf(df_diff, 'Close_diff', 'Differenced Data')
    else:
        # Plot ACF and PACF for original data
        plot_acf_pacf(df, 'Close', 'Original Data')