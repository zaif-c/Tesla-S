import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_features(file_path: str = 'data/processed/tesla_processed_data.csv') -> pd.DataFrame:
    """
    Load the engineered features.
    
    Args:
        file_path (str): Path to the features CSV file
        
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    return df

def plot_time_series(df: pd.DataFrame) -> None:
    """
    Plot the closing price and moving averages.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.plot(df.index, df['MA5'], label='MA5')
    plt.plot(df.index, df['MA10'], label='MA10')
    plt.plot(df.index, df['MA20'], label='MA20')
    plt.title('Tesla Stock Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/eda_plots/time_series_plot.png')
    plt.close()

def plot_returns_histogram(df: pd.DataFrame) -> None:
    """
    Plot histograms of daily and monthly returns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.histplot(df['Daily_Return'].dropna(), kde=True, ax=axes[0])
    axes[0].set_title('Daily Returns')
    
    sns.histplot(df['Monthly_Return'].dropna(), kde=True, ax=axes[1])
    axes[1].set_title('Monthly Returns')
    
    plt.tight_layout()
    plt.savefig('outputs/eda_plots/returns_histogram.png')
    plt.close()

def plot_rolling_volatility(df: pd.DataFrame) -> None:
    """
    Plot rolling volatility over time.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Rolling_Volatility'])
    plt.title('Rolling Volatility (10-day)')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.savefig('outputs/eda_plots/rolling_volatility_plot.png')
    plt.close()

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Plot correlation heatmap of the features.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('outputs/eda_plots/correlation_heatmap.png')
    plt.close()

if __name__ == "__main__":
    # Load the engineered features
    df = load_features()
    
    # Create visualizations
    plot_time_series(df)
    plot_returns_histogram(df)
    plot_rolling_volatility(df)
    plot_correlation_heatmap(df)
    
    print("EDA visualizations saved to outputs/") 