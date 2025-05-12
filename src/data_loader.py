import pandas as pd
import numpy as np
from pathlib import Path

def load_tesla_data(file_path: str = 'data/tesla_raw_data.csv') -> pd.DataFrame:
    """
    Load Tesla stock data from CSV file and perform initial preprocessing.
    
    Args:
        file_path (str): Path to the CSV file containing Tesla stock data
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with datetime index
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert Date column to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Sort by date
    df.sort_index(inplace=True)
    
    return df

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for missing values in the DataFrame and return a summary.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: Summary of missing values
    """
    # Calculate missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    # Create summary DataFrame
    missing_summary = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    
    return missing_summary

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # For stock data, we'll use forward fill for missing values
    # This assumes missing values are due to non-trading days
    df_clean.ffill(inplace=True)
    
    # If there are still any missing values at the start of the dataset,
    # use backward fill
    df_clean.bfill(inplace=True)
    
    return df_clean

if __name__ == "__main__":
    # Load the data
    df = load_tesla_data()
    
    # Print basic information
    print("\nDataset Info:")
    print("-" * 50)
    print(f"Date Range: {df.index.min()} to {df.index.max()}")
    print(f"Number of records: {len(df)}")
    print("\nColumns:")
    print(df.columns.tolist())
    
    # Check for missing values
    print("\nMissing Values Summary:")
    print("-" * 50)
    missing_summary = check_missing_values(df)
    print(missing_summary)
    
    # Handle missing values
    df_clean = handle_missing_values(df)
    
    # Verify no missing values remain
    print("\nMissing Values After Cleaning:")
    print("-" * 50)
    print(df_clean.isnull().sum())
    
    # Save the cleaned data
    output_path = Path('data/processed/tesla_cleaned_data.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path)
    print(f"\nCleaned data saved to: {output_path}") 