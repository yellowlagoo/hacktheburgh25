import requests
from bs4 import BeautifulSoup
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from typing import List, Tuple, Dict, Optional, Any
import logging
import os
import json
from datetime import datetime
from pathlib import Path
import pickle
import hashlib
from PIL import Image
from io import BytesIO
from functools import wraps
import cv2
from social_media_scraper import scrape_social_media_images

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("market_sentiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create cache directories
CACHE_DIR = Path("cache")
IMAGE_CACHE = CACHE_DIR / "images"
DATA_CACHE = CACHE_DIR / "data"
RESULTS_DIR = Path("results")

for directory in [CACHE_DIR, IMAGE_CACHE, DATA_CACHE, RESULTS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

def retry_decorator(max_retries=3, delay=2):
    """
    Decorator for retrying functions that might fail temporarily
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.RequestException, 
                        ConnectionError, 
                        TimeoutError) as e:
                    retries += 1
                    wait_time = delay * (2 ** (retries - 1))  # Exponential backoff
                    logger.warning(
                        f"Retry {retries}/{max_retries} for {func.__name__} after error: {str(e)}. "
                        f"Waiting {wait_time}s"
                    )
                    time.sleep(wait_time)
            
            # If we get here, all retries failed
            logger.error(f"All {max_retries} retries failed for {func.__name__}")
            raise Exception(f"Function {func.__name__} failed after {max_retries} retries")
        return wrapper
    return decorator



class ConsumerDataAnalyzer:
    def __init__(self, coffee_data_path: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        self.coffee_data_path = coffee_data_path
        self.start_date = start_date
        self.end_date = end_date
        
        # Load and validate data
        self.coffee_data = self._load_and_validate_coffee_data(coffee_data_path)

    def _load_and_validate_coffee_data(self, data_path: str) -> pd.DataFrame:
        """Load, validate, and clean the coffee data for the specified time period"""
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            # Check if required column exists (column name may be APU0000717311)
            value_column = 'APU0000717311' if 'APU0000717311' in df.columns else df.columns[1]
            
            # Convert date column to datetime
            date_column = 'observation_date' if 'observation_date' in df.columns else df.columns[0]
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Remove rows with missing values
            df = df.dropna(subset=[value_column])
            
            # Filter by date range if specified
            if self.start_date and self.end_date:
                start = pd.to_datetime(self.start_date)
                end = pd.to_datetime(self.end_date)
                filtered_data = df[(df[date_column] >= start) & (df[date_column] <= end)].copy()
                
                if len(filtered_data) == 0:
                    logger.warning(f"No data found for period {self.start_date} to {self.end_date}. Using all available data.")
                    filtered_data = df.copy()
            elif self.start_date:
                start = pd.to_datetime(self.start_date)
                filtered_data = df[df[date_column] >= start].copy()
                
                if len(filtered_data) == 0:
                    logger.warning(f"No data found after {self.start_date}. Using all available data.")
                    filtered_data = df.copy()
            elif self.end_date:
                end = pd.to_datetime(self.end_date)
                filtered_data = df[df[date_column] <= end].copy()
                
                if len(filtered_data) == 0:
                    logger.warning(f"No data found before {self.end_date}. Using all available data.")
                    filtered_data = df.copy()
            else:
                # If no date range specified, use most recent 12 months
                df = df.sort_values(by=date_column, ascending=False)
                filtered_data = df.head(12).copy()
            
            # Add spending column for consistency with the rest of the code
            filtered_data['spending'] = filtered_data[value_column]
            
            date_range_str = ""
            if self.start_date and self.end_date:
                date_range_str = f" for period {self.start_date} to {self.end_date}"
            elif self.start_date:
                date_range_str = f" since {self.start_date}"
            elif self.end_date:
                date_range_str = f" until {self.end_date}"
            
            logger.info(f"Successfully loaded and cleaned coffee data{date_range_str}: {len(filtered_data)} records")
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error loading or validating coffee data from {data_path}: {str(e)}")
            raise

    def process_consumer_data(self) -> float:
        """
        Process coffee spending data
        Returns normalized value
        """
        # Calculate average monthly coffee spending
        avg_coffee = self.coffee_data['spending'].mean()
        
        # Normalize value between 0 and 1
        norm_coffee = (avg_coffee - self.coffee_data['spending'].min()) / \
                     (self.coffee_data['spending'].max() - self.coffee_data['spending'].min())
        
        logger.info(f"Normalized coffee spending: {norm_coffee:.4f}")
        
        return norm_coffee

class MarketSentimentPredictor:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = self._load_model() if model_path else RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False if model_path is None else True

    def _load_model(self) -> Any:
        """Load trained model from disk"""
        try:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Loaded model from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {str(e)}")
            return RandomForestRegressor(n_estimators=100, random_state=42)

    def _save_model(self, path: str) -> None:
        """Save model to disk"""
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Saved model to {path}")
        except Exception as e:
            logger.error(f"Error saving model to {path}: {str(e)}")

    def train_model(self, X: np.array, y: np.array) -> Dict[str, float]:
        """
        Train the sentiment prediction model with validation
        """
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training model with {X_train.shape[0]} samples, testing with {X_test.shape[0]} samples")
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        metrics = {
            'rmse': rmse,
            'mae': mae
        }
        
        logger.info(f"Model evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}")
        
        # Save the model
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = model_dir / f"sentiment_model_{timestamp}.pkl"
        self._save_model(str(model_path))
        
        return metrics

    def predict_sentiment(self, features: np.array) -> float:
        """
        Predict market sentiment based on input features
        Returns a value between 0 and 1
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call train_model() first.")
            
        prediction = self.model.predict(features.reshape(1, -1))[0]
        result = np.clip(prediction, 0, 1)
        logger.info(f"Predicted sentiment: {result:.4f}")
        return result

def analyze_market_sentiment(
    search_query: str,
    num_images: int,
    coffee_data_path: str,
    platforms: List[str] = None,
    model_path: Optional[str] = None,
    output_file: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    historical_mode: bool = False
) -> Dict[str, Any]:
    """
    Main function to analyze market sentiment using skirt lengths and coffee data
    
    Args:
        search_query: Query for social media images
        num_images: Number of images to analyze
        coffee_data_path: Path to coffee price data CSV
        platforms: List of social media platforms to scrape
        model_path: Path to pre-trained model (optional)
        output_file: File to save results (optional)
        start_date: Start date for analysis (format: YYYY-MM-DD)
        end_date: End date for analysis (format: YYYY-MM-DD)
        historical_mode: If True, uses coffee data only for historical analysis
        
    Returns:
        Dictionary with results and metadata
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Initialize components
        skirt_analyzer = SkirtLengthAnalyzer(model_path)
        consumer_analyzer = ConsumerDataAnalyzer(
            coffee_data_path, 
            start_date=start_date, 
            end_date=end_date
        )
        predictor = MarketSentimentPredictor(model_path)

        # For historical mode, we only use coffee data
        if historical_mode and (start_date or end_date):
            logger.info(f"Running in historical mode for period: {start_date or 'beginning'} to {end_date or 'present'}")
            
            # Process coffee data for the time period
            coffee_score = consumer_analyzer.process_consumer_data()
           
            
            # Prepare features
            features = np.array(coffee_score)
        else:
            logger.info("Step : Processing consumer data")
            coffee_score = consumer_analyzer.process_consumer_data()

            features = np.array([coffee_score])
        
        # Mock training data if needed (in a real scenario, this would be historical data)
        if not predictor.is_trained and model_path is None:
            logger.info("Generating synthetic training data for model training")
            # Generate synthetic data for demonstration
            n_samples = 100
            X_synth = np.random.rand(n_samples, 1)  # 1 feature now: just coffee
            # Synthetic formula: higher coffee spending correlates with bullish sentiment
            y_synth = X_synth[:, 0]  # Coffee spending directly proportional to sentiment
            y_synth = np.clip(y_synth + np.random.normal(0, 0.1, n_samples), 0, 1)
            
            logger.info("Step 4: Training model with synthetic data")
            metrics = predictor.train_model(X_synth, y_synth)
        else:
            metrics = {"info": "Using pre-trained model"}
        
        # Predict sentiment
        logger.info("Step 5: Predicting market sentiment")
        sentiment_score = predictor.predict_sentiment(features)
        
        # Prepare time period description
        time_period = "recent data"
        if start_date and end_date:
            time_period = f"{start_date} to {end_date}"
        elif start_date:
            time_period = f"since {start_date}"
        elif end_date:
            time_period = f"until {end_date}"
        
        # Prepare results
        results = {
            "timestamp": timestamp,
            "sentiment_score": float(sentiment_score),
            "time_period": time_period,
            "features": {
                "coffee_spending": float(coffee_score)
            },
            "metadata": {
                "search_query": search_query,
                "historical_mode": historical_mode,
                "start_date": start_date,
                "end_date": end_date,
                "num_images_requested": 0 if historical_mode else num_images,
                "num_images_processed": 0 if historical_mode else len(image_paths),
                "processing_time_seconds": time.time() - start_time,
                "model_metrics": metrics
            }
        }
        
        # Save results to file if specified
        if output_file:
            output_path = RESULTS_DIR / output_file
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        
        # Also save to CSV for time series analysis
        csv_path = RESULTS_DIR / "sentiment_history.csv"
        csv_exists = csv_path.exists()
        
        with open(csv_path, 'a') as f:
            if not csv_exists:
                f.write("timestamp,time_period,sentiment_score,avg_skirt_length,coffee_spending\n")
            f.write(f"{timestamp},{time_period},{sentiment_score},{avg_skirt_length},{coffee_score}\n")
        
        logger.info(f"Analysis complete. Sentiment score for {time_period}: {sentiment_score:.4f}")
        return results

    except Exception as e:
        error_msg = f"Error in market sentiment analysis: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "error": error_msg,
            "timestamp": timestamp,
            "success": False
        }

def analyze_sentiment_time_series(
    coffee_data_path: str,
    output_file: str = "sentiment_timeline.json",
    period: str = "yearly"
) -> Dict[str, Any]:
    """
    Analyze sentiment over multiple time periods
    
    Args:
        coffee_data_path: Path to coffee price data
        output_file: File to save results
        period: Aggregation period ('yearly', 'quarterly', 'monthly')
        
    Returns:
        Dictionary with time series results
    """
    # Load all coffee data
    df = pd.read_csv(coffee_data_path)
    date_column = 'observation_date' if 'observation_date' in df.columns else df.columns[0]
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Sort by date
    df = df.sort_values(by=date_column)
    
    # Determine time periods based on the period parameter
    if period == 'yearly':
        # Group by year
        df['period'] = df[date_column].dt.year
    elif period == 'quarterly':
        # Group by year and quarter
        df['period'] = df[date_column].dt.year.astype(str) + 'Q' + df[date_column].dt.quarter.astype(str)
    elif period == 'monthly':
        # Group by year and month
        df['period'] = df[date_column].dt.year.astype(str) + '-' + df[date_column].dt.month.astype(str).str.zfill(2)
    else:
        raise ValueError(f"Invalid period: {period}. Use 'yearly', 'quarterly', or 'monthly'.")
    
    # Get unique periods
    periods = df['period'].unique()
    
    # Analyze each period
    results = []
    for period_value in periods:
        period_data = df[df['period'] == period_value]
        
        if len(period_data) <= 1:
            # Skip periods with insufficient data
            continue
            
        # Get date range for this period
        start_date = period_data[date_column].min().strftime('%Y-%m-%d')
        end_date = period_data[date_column].max().strftime('%Y-%m-%d')
        
        # Run analysis for this period
        period_result = analyze_market_sentiment(
            search_query="skirts fashion trend",
            num_images=0,  # Not used in historical mode
            coffee_data_path=coffee_data_path,
            start_date=start_date,
            end_date=end_date,
            historical_mode=True
        )
        
        # Store the result
        if "error" not in period_result:
            results.append({
                "period": str(period_value),
                "start_date": start_date,
                "end_date": end_date,
                "sentiment_score": period_result["sentiment_score"],
                "coffee_spending": period_result["features"]["coffee_spending"]
            })
    
    # Save the time series results
    final_results = {
        "period_type": period,
        "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "results": results
    }
    
    output_path = RESULTS_DIR / output_file
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"Time series analysis complete. Results saved to {output_path}")
    return final_results

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Use the provided coffee data file
    coffee_file = Path("APU0000717311.csv")
    if not coffee_file.exists():
        logger.error(f"Coffee data file {coffee_file} not found!")
        print(f"Error: Coffee data file {coffee_file} not found!")
        exit(1)
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Market Sentiment Analyzer')
    parser.add_argument('--mode', choices=['current', 'historical', 'timeline'], 
                        default='current', help='Analysis mode')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--period', choices=['yearly', 'quarterly', 'monthly'], 
                        default='yearly', help='Period for timeline analysis')
    parser.add_argument('--images', type=int, default=10, 
                        help='Number of images to analyze (current mode only)')
    
    args = parser.parse_args()
    
    if args.mode == 'current':
        # Standard analysis with current social media images
        results = analyze_market_sentiment(
            search_query="skirts fashion trend",
            num_images=args.images,
            coffee_data_path=str(coffee_file),
            platforms=['pinterest'],
            start_date=args.start_date,
            end_date=args.end_date,
            output_file=f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        # Display results
        if "error" in results:
            print(f"\nError: {results['error']}")
        else:
            print(f"\nMarket Sentiment Analysis Results:")
            print(f"Time Period: {results['time_period']}")
            print(f"Sentiment Score: {results['sentiment_score']:.4f}")
            print(f"Based on:")
            print(f"  - Average Skirt Length: {results['features']['avg_skirt_length']:.4f}")
            print(f"  - Coffee Spending Index: {results['features']['coffee_spending']:.4f}")
            
    elif args.mode == 'historical':
        # Historical analysis based on coffee data only
        if not args.start_date and not args.end_date:
            print("Error: Historical mode requires at least one of --start-date or --end-date")
            exit(1)
            
        results = analyze_market_sentiment(
            search_query="skirts fashion trend",
            num_images=0,  # Not used in historical mode
            coffee_data_path=str(coffee_file),
            start_date=args.start_date,
            end_date=args.end_date,
            historical_mode=True,
            output_file=f"historical_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        # Display results
        if "error" in results:
            print(f"\nError: {results['error']}")
        else:
            print(f"\nHistorical Market Sentiment Analysis:")
            print(f"Time Period: {results['time_period']}")
            print(f"Sentiment Score: {results['sentiment_score']:.4f}")
            print(f"Based primarily on Coffee Spending Index: {results['features']['coffee_spending']:.4f}")
            
    elif args.mode == 'timeline':
        # Time series analysis across multiple periods
        results = analyze_sentiment_time_series(
            coffee_data_path=str(coffee_file),
            period=args.period,
            output_file=f"sentiment_timeline_{args.period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        # Display summary
        print(f"\nTime Series Analysis Complete")
        print(f"Analyzed {len(results['results'])} {args.period} periods")
        
        # Display a few sample periods
        if results['results']:
            print("\nSample Results:")
            for period in results['results'][-5:]:  # Show last 5 periods
                print(f"Period: {period['period']}, Sentiment: {period['sentiment_score']:.4f}")