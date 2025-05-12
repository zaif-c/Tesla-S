import pandas as pd
import numpy as np
from pathlib import Path

def load_cleaned_data(file_path: str = 'data/processed/tesla_cleaned_data.csv') -> pd.DataFrame:
    """
    Load the cleaned Tesla stock data.
    
    Args:
        file_path (str): Path to the cleaned CSV file
        
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from the cleaned DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    # Create a copy to avoid modifying the original
    df_features = df.copy()
    
    # 1. Moving Averages
    df_features['MA5'] = df_features['Close'].rolling(window=5).mean()
    df_features['MA10'] = df_features['Close'].rolling(window=10).mean()
    df_features['MA20'] = df_features['Close'].rolling(window=20).mean()
    
    # 2. Rolling Volatility (10-day standard deviation of returns)
    df_features['Daily_Return'] = df_features['Close'].pct_change()
    df_features['Rolling_Volatility'] = df_features['Daily_Return'].rolling(window=10).std()
    
    # 3. Monthly Returns (approx. 21 trading days)
    df_features['Monthly_Return'] = df_features['Close'].pct_change(periods=21)
    
    # 4. Normalize/Standardize features (optional)
    # For now, we'll skip normalization, but you can add it later if needed
    
    return df_features

if __name__ == "__main__":
    # Load the cleaned data
    df = load_cleaned_data()
    
    # Engineer features
    df_features = engineer_features(df)
    
    # Print basic information
    print("\nEngineered Features Info:")
    print("-" * 50)
    print(f"Number of records: {len(df_features)}")
    print("\nColumns:")
    print(df_features.columns.tolist())
    
    # Save the engineered features
    output_path = Path('data/processed/tesla_processed_data.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path)
    print(f"\nEngineered features saved to: {output_path}") 