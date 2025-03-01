import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import requests
from datetime import datetime, timedelta
import os
from pathlib import Path
import logging
import json
from tqdm import tqdm
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('coffee_weather_predictor')

# Create directories
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

class CoffeePriceLoader:
    """Class to load and preprocess historical coffee price data"""
    
    def __init__(self, file_path="APU0000717311.csv"):
        self.file_path = file_path
        
    def load_data(self):
        """Load coffee price data from CSV file"""
        try:
            # Load the CSV file
            df = pd.read_csv(self.file_path)
            
            # Identify date and price columns
            date_col = df.columns[0] if 'date' in df.columns[0].lower() else 'observation_date'
            price_col = df.columns[1] if len(df.columns) > 1 else 'APU0000717311'
            
            # Rename columns for consistency
            df = df.rename(columns={date_col: 'date', price_col: 'coffee_price'})
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Sort by date
            df = df.sort_values('date')
            
            # Check for missing values
            missing_values = df['coffee_price'].isnull().sum()
            if missing_values > 0:
                logger.info(f"Found {missing_values} missing price values. Interpolating...")
                df['coffee_price'] = df['coffee_price'].interpolate(method='linear')
            
            # Add month and year columns for easier analysis
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            
            logger.info(f"Loaded coffee price data with {len(df)} records from {df['date'].min()} to {df['date'].max()}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading coffee price data: {str(e)}")
            raise

class WeatherDataFetcher:
    """Class to fetch historical weather data for coffee-producing countries"""
    
    def __init__(self, api_key=None):
        # Default API key (replace with your own)
        self.api_key = api_key or os.environ.get('WEATHER_API_KEY', '')
        
        # Major coffee-producing regions with coordinates
        self.coffee_regions = {
            'Brazil': [
                {'name': 'Minas Gerais', 'lat': -19.9167, 'lon': -43.9345},
                {'name': 'Sao Paulo', 'lat': -23.5505, 'lon': -46.6333},
                {'name': 'Espirito Santo', 'lat': -20.2976, 'lon': -40.2958}
            ],
            'Vietnam': [
                {'name': 'Central Highlands', 'lat': 12.6992, 'lon': 108.0765},
                {'name': 'Dak Lak', 'lat': 12.6667, 'lon': 108.0500}
            ],
            'Colombia': [
                {'name': 'Huila', 'lat': 2.5359, 'lon': -75.5277},
                {'name': 'Nariño', 'lat': 1.2136, 'lon': -77.2811},
                {'name': 'Antioquia', 'lat': 7.1986, 'lon': -75.3412}
            ]
        }
        
        # Cache file path
        self.cache_file = DATA_DIR / "weather_data_cache.json"
        
        # Create cache if it doesn't exist
        if not self.cache_file.exists():
            with open(self.cache_file, 'w') as f:
                json.dump({}, f)
                
    def _load_cache(self):
        """Load weather data cache"""
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except:
            return {}
            
    def _save_cache(self, cache):
        """Save weather data to cache"""
        with open(self.cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
    
    def fetch_weather_data(self, start_date, end_date):
        """
        Fetch or generate historical weather data for coffee-producing regions
        
        Note: For demonstration purposes, this function generates synthetic weather 
        data rather than making actual API calls, which would require paid subscriptions
        for historical data.
        """
        logger.info(f"Fetching/generating weather data from {start_date} to {end_date}")
        
        # Load cache
        cache = self._load_cache()
        cache_key = f"{start_date}_{end_date}"
        
        # Check if data is already cached
        if cache_key in cache:
            logger.info("Using cached weather data")
            return pd.DataFrame(cache[cache_key])
        
        # Convert dates to datetime objects
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Generate date range
        date_range = pd.date_range(start=start, end=end, freq='MS')  # Monthly data
        
        # Prepare data structure
        weather_data = []
        
        # Generate synthetic weather data for each region and date
        for country, regions in self.coffee_regions.items():
            for region in regions:
                # Set random seed based on region for consistency
                region_name = f"{country}_{region['name']}"
                seed_value = abs(hash(region_name)) % (2**32 - 1)
                np.random.seed(seed_value)
                
                for date in date_range:
                    # Base values with seasonal variations
                    month = date.month
                    is_southern = region['lat'] < 0  # Southern hemisphere
                    
                    # Adjust season based on hemisphere
                    if is_southern:
                        # Southern hemisphere seasons are opposite
                        season_factor = np.sin(np.pi * (month - 6) / 6)
                    else:
                        season_factor = np.sin(np.pi * month / 6)
                    
                    # Generate weather data with seasonal patterns
                    # Temperature in Celsius
                    base_temp = 25 if is_southern else 28
                    temp = base_temp + 4 * season_factor + np.random.normal(0, 1)
                    
                    # Rainfall in mm (higher in rainy seasons)
                    base_rain = 100 + 100 * season_factor
                    rainfall = max(0, base_rain + np.random.normal(0, 30))
                    
                    # Humidity percentage
                    humidity = 70 + 10 * season_factor + np.random.normal(0, 5)
                    humidity = max(40, min(95, humidity))
                    
                    # Extreme weather flag (rare events)
                    extreme_weather = 1 if np.random.random() < 0.05 else 0
                    
                    # Add yearly patterns (e.g., El Niño cycles every ~3-7 years)
                    year = date.year
                    el_nino_cycle = np.sin(2 * np.pi * (year - 1980) / 5)
                    
                    # Adjust temperature and rainfall for El Niño cycles
                    temp += el_nino_cycle * 1.5
                    rainfall += el_nino_cycle * 30
                    
                    # Store the data
                    weather_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'country': country,
                        'region': region['name'],
                        'temperature': round(temp, 1),
                        'rainfall': round(rainfall, 1),
                        'humidity': round(humidity, 1),
                        'extreme_weather': extreme_weather
                    })
        
        # Convert to DataFrame
        df_weather = pd.DataFrame(weather_data)
        
        # Cache the results
        cache[cache_key] = df_weather.to_dict('records')
        self._save_cache(cache)
        
        return df_weather

class CoffeeWeatherModel:
    """Class to build and train the coffee price prediction model"""
    
    def __init__(self):
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def preprocess_data(self, coffee_data, weather_data):
        """Merge and preprocess coffee price and weather data"""
        logger.info("Preprocessing and merging datasets")
        
        # Ensure date column is in datetime format
        coffee_data['date'] = pd.to_datetime(coffee_data['date'])
        weather_data['date'] = pd.to_datetime(weather_data['date'])
        
        # Group weather data by date (averaging across regions)
        weather_agg = weather_data.groupby(['date', 'country']).agg({
            'temperature': 'mean',
            'rainfall': 'mean',
            'humidity': 'mean',
            'extreme_weather': 'max'  # If any region had extreme weather
        }).reset_index()
        
        # Pivot to wide format to have each country's weather as separate features
        weather_wide = pd.pivot_table(
            weather_agg, 
            values=['temperature', 'rainfall', 'humidity', 'extreme_weather'],
            index='date',
            columns='country'
        )
        
        # Flatten the column names
        weather_wide.columns = ['_'.join(col).strip() for col in weather_wide.columns]
        
        # Reset index to make date a column again
        weather_wide = weather_wide.reset_index()
        
        # Merge with coffee prices by date
        # For each coffee price date, find the closest weather date (not exceeding)
        merged_data = pd.DataFrame()
        
        for idx, row in coffee_data.iterrows():
            coffee_date = row['date']
            # Find weather data for the most recent month
            closest_weather_date = weather_wide[weather_wide['date'] <= coffee_date]['date'].max()
            
            if pd.notna(closest_weather_date):
                weather_row = weather_wide[weather_wide['date'] == closest_weather_date].iloc[0].to_dict()
                merged_row = {**row.to_dict(), **weather_row}
                merged_data = pd.concat([merged_data, pd.DataFrame([merged_row])], ignore_index=True)
        
        # Drop any rows with missing data
        merged_data = merged_data.dropna()
        
        logger.info(f"Merged dataset contains {len(merged_data)} records")
        
        # Create lag features (previous month's weather)
        feature_cols = [col for col in merged_data.columns if any(country in col for country in ['Brazil', 'Vietnam', 'Colombia'])]
        
        for col in feature_cols:
            merged_data[f'{col}_lag1'] = merged_data[col].shift(1)
            merged_data[f'{col}_lag3'] = merged_data[col].shift(3)
            merged_data[f'{col}_lag6'] = merged_data[col].shift(6)
        
        # Drop rows with NaN values from lag creation
        merged_data = merged_data.dropna()
        
        # Add month and quarter as cyclical features
        merged_data['month_sin'] = np.sin(2 * np.pi * merged_data['month'] / 12)
        merged_data['month_cos'] = np.cos(2 * np.pi * merged_data['month'] / 12)
        
        # Add trend feature
        merged_data['trend'] = range(len(merged_data))
        
        return merged_data
    
    def train(self, data, model_type='linear'):
        """Train the prediction model"""
        logger.info(f"Training {model_type} regression model")
        
        # Identify feature columns (all weather-related columns and time features)
        feature_cols = [col for col in data.columns if any(x in col for x in 
                       ['Brazil', 'Vietnam', 'Colombia', 'month_sin', 'month_cos', 'trend'])]
        
        X = data[feature_cols]
        y = data['coffee_price']
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # Choose model type
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Evaluate the model
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        logger.info(f"Train RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R²: {train_r2:.4f}")
        logger.info(f"Test RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R²: {test_r2:.4f}")
        
        # Store test data for visualization
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_scaled = X_test_scaled
        self.feature_cols = feature_cols
        
        # If it's a linear model, analyze coefficients
        if model_type == 'linear':
            self._analyze_coefficients()
            
        # Return metrics for reference
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    
    def _analyze_coefficients(self):
        """Analyze feature importance for linear model"""
        if not isinstance(self.model, LinearRegression):
            return
            
        # Get feature importances
        coefficients = pd.DataFrame({
            'Feature': self.feature_cols,
            'Coefficient': self.model.coef_
        })
        
        # Sort by absolute value of coefficient
        coefficients['Abs_Coefficient'] = abs(coefficients['Coefficient'])
        coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)
        
        logger.info("Feature importance (top 10 coefficients):")
        for i, row in coefficients.head(10).iterrows():
            logger.info(f"{row['Feature']}: {row['Coefficient']:.4f}")
    
    def predict(self, weather_data):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        # Ensure the input has all required features
        missing_features = set(self.feature_cols) - set(weather_data.columns)
        if missing_features:
            raise ValueError(f"Missing features in input data: {missing_features}")
        
        # Select only the required features
        X_pred = weather_data[self.feature_cols]
        
        # Scale the features
        X_pred_scaled = self.scaler_X.transform(X_pred)
        
        # Make prediction
        predictions = self.model.predict(X_pred_scaled)
        
        return predictions
    
    def save_model(self, file_path=None):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No trained model to save")
            
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = MODELS_DIR / f"coffee_price_model_{timestamp}.pkl"
        
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler_X': self.scaler_X,
                'feature_cols': self.feature_cols
            }, f)
            
        logger.info(f"Model saved to {file_path}")
    
    def load_model(self, file_path):
        """Load a trained model from file"""
        import pickle
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.scaler_X = model_data['scaler_X']
        self.feature_cols = model_data['feature_cols']
        
        logger.info(f"Model loaded from {file_path}")
    
    def visualize_results(self):
        """Visualize model predictions vs actual values"""
        if self.model is None or not hasattr(self, 'X_test_scaled'):
            raise ValueError("Model has not been trained or test data is missing")
        
        # Make predictions on test data
        y_pred = self.model.predict(self.X_test_scaled)
        
        # Create a DataFrame with actual and predicted values
        results_df = pd.DataFrame({
            'Actual': self.y_test,
            'Predicted': y_pred
        }).reset_index()
        
        # Create visualizations directory
        viz_dir = RESULTS_DIR / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Scatter plot of actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['Actual'], results_df['Predicted'], alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
        max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Actual Coffee Price')
        plt.ylabel('Predicted Coffee Price')
        plt.title('Actual vs Predicted Coffee Prices')
        plt.grid(True, alpha=0.3)
        plt.savefig(viz_dir / "actual_vs_predicted.png")
        
        # 2. Predictions over time
        # Add date info to results
        results_df = results_df.merge(
            self.X_test.reset_index()[['index', 'date']],
            left_on='index',
            right_on='index'
        )
        results_df = results_df.sort_values('date')
        
        plt.figure(figsize=(12, 6))
        plt.plot(results_df['date'], results_df['Actual'], label='Actual', marker='o', alpha=0.7)
        plt.plot(results_df['date'], results_df['Predicted'], label='Predicted', marker='x', alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('Coffee Price')
        plt.title('Coffee Price Predictions Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(viz_dir / "predictions_over_time.png")
        
        # 3. Residual plot
        results_df['Residual'] = results_df['Actual'] - results_df['Predicted']
        
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['Predicted'], results_df['Residual'], alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Coffee Price')
        plt.ylabel('Residual')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        plt.savefig(viz_dir / "residual_plot.png")
        
        # 4. Feature importance plot (for random forest)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title('Feature Importances')
            plt.bar(range(len(indices)), importances[indices], align='center')
            plt.xticks(range(len(indices)), [self.feature_cols[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(viz_dir / "feature_importance.png")
        
        logger.info(f"Visualizations saved to {viz_dir}")
        
        return viz_dir

def generate_future_weather(months_ahead=6):
    """Generate synthetic weather data for future predictions"""
    # Create weather data fetcher
    weather_fetcher = WeatherDataFetcher()
    
    # Get the current date
    current_date = datetime.now()
    
    # Generate future dates
    future_start = current_date.replace(day=1)
    future_end = (future_start + timedelta(days=31*months_ahead)).replace(day=1)
    
    # Fetch/generate weather data
    future_weather = weather_fetcher.fetch_weather_data(
        future_start.strftime('%Y-%m-%d'),
        future_end.strftime('%Y-%m-%d')
    )
    
    return future_weather

def prepare_future_prediction_data(future_weather, historical_data):
    """Prepare future weather data for prediction by adding required features"""
    # Group weather data by date (averaging across regions)
    weather_agg = future_weather.groupby(['date', 'country']).agg({
        'temperature': 'mean',
        'rainfall': 'mean',
        'humidity': 'mean',
        'extreme_weather': 'max'
    }).reset_index()
    
    # Pivot to wide format
    weather_wide = pd.pivot_table(
        weather_agg, 
        values=['temperature', 'rainfall', 'humidity', 'extreme_weather'],
        index='date',
        columns='country'
    ).reset_index()
    
    # Flatten column names
    weather_wide.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in weather_wide.columns]
    
    # Convert to datetime
    weather_wide['date'] = pd.to_datetime(weather_wide['date'])
    
    # Add month column
    weather_wide['month'] = weather_wide['date'].dt.month
    
    # Add cyclical month features
    weather_wide['month_sin'] = np.sin(2 * np.pi * weather_wide['month'] / 12)
    weather_wide['month_cos'] = np.cos(2 * np.pi * weather_wide['month'] / 12)
    
    # Add trend by extending from historical data
    last_trend = len(historical_data) - 1
    weather_wide['trend'] = range(last_trend + 1, last_trend + 1 + len(weather_wide))
    
    # Create lag features from historical + current data
    # Combine historical and future data temporarily
    historical_cols = [col for col in historical_data.columns 
                      if any(country in col for country in ['Brazil', 'Vietnam', 'Colombia'])]
    
    # Get the last few records from historical data
    last_records = historical_data[['date'] + historical_cols].tail(6).copy()
    
    # Combine with future data
    combined = pd.concat([last_records, weather_wide], ignore_index=True)
    combined = combined.sort_values('date')
    
    # Create lag features
    feature_cols = [col for col in combined.columns 
                   if any(country in col for country in ['Brazil', 'Vietnam', 'Colombia'])
                   and not col.endswith(('_lag1', '_lag3', '_lag6'))]
    
    for col in feature_cols:
        combined[f'{col}_lag1'] = combined[col].shift(1)
        combined[f'{col}_lag3'] = combined[col].shift(3)
        combined[f'{col}_lag6'] = combined[col].shift(6)
    
    # Keep only future data
    future_data = combined[combined['date'] >= weather_wide['date'].min()].copy()
    
    return future_data

def main():
    """Main function to run the coffee price prediction model"""
    # Load coffee price data
    coffee_loader = CoffeePriceLoader()
    coffee_data = coffee_loader.load_data()
    
    # Get time range for weather data from coffee data
    start_date = coffee_data['date'].min().strftime('%Y-%m-%d')
    end_date = coffee_data['date'].max().strftime('%Y-%m-%d')
    
    # Fetch/generate weather data
    weather_fetcher = WeatherDataFetcher()
    weather_data = weather_fetcher.fetch_weather_data(start_date, end_date)
    
    # Create and train model
    model = CoffeeWeatherModel()
    
    # Preprocess and merge data
    merged_data = model.preprocess_data(coffee_data, weather_data)
    
    # Train linear regression model
    logger.info("Training linear regression model...")
    linear_metrics = model.train(merged_data, model_type='linear')
    
    # Save the model
    model.save_model()
    
    # Visualize results
    viz_dir = model.visualize_results()
    
    # Train random forest model for comparison
    logger.info("\nTraining random forest model for comparison...")
    rf_model = CoffeeWeatherModel()
    rf_metrics = rf_model.train(merged_data, model_type='random_forest')
    
    # Save the random forest model
    rf_model.save_model(MODELS_DIR / f"coffee_price_rf_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
    
    # Visualize random forest results
    rf_viz_dir = rf_model.visualize_results()
    
    # Generate future predictions
    logger.info("\nGenerating future predictions for next 6 months...")
    
    future_weather = generate_future_weather(months_ahead=6)
    future_data = prepare_future_prediction_data(future_weather, merged_data)
    
    # Make predictions using linear model
    future_predictions_linear = model.predict(future_data)
    
    # Make predictions using random forest model
    future_predictions_rf = rf_model.predict(future_data)
    
    # Create DataFrame with predictions
    predictions_df = pd.DataFrame({
        'date': future_data['date'],
        'linear_prediction': future_predictions_linear,
        'random_forest_prediction': future_predictions_rf
    })
    
    # Save predictions
    predictions_file = RESULTS_DIR / f"coffee_price_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    predictions_df.to_csv(predictions_file, index=False)
    
    # Display future predictions
    logger.info("\nFuture Coffee Price Predictions:")
    logger.info(predictions_df)
    
    # Visualize future predictions
    plt.figure(figsize=(12, 6))
    plt.plot(predictions_df['date'], predictions_df['linear_prediction'], label='Linear Model', marker='o')
    plt.plot(predictions_df['date'], predictions_df['random_forest_prediction'], label='Random Forest Model', marker='x')
    plt.xlabel('Date')
    plt.ylabel('Predicted Coffee Price')
    plt.title('Future Coffee Price Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "future_predictions.png")
    
    logger.info(f"\nFuture predictions saved to {predictions_file}")
    logger.info(f"Future prediction chart saved to {RESULTS_DIR / 'future_predictions.png'}")
    
    # Print summary
    logger.info("\nModel Performance Summary:")
    logger.info(f"Linear Regression - Test R²: {linear_metrics['test_r2']:.4f}, RMSE: {linear_metrics['test_rmse']:.2f}")
    logger.info(f"Random Forest - Test R²: {rf_metrics['test_r2']:.4f}, RMSE: {rf_metrics['test_rmse']:.2f}")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Coffee Price Prediction based on Weather Data")
    parser.add_argument('--coffee-file', type=str, default="APU0000717311.csv",
                       help="Path to coffee price data CSV file")
    parser.add_argument('--months-ahead', type=int, default=6,
                       help="Number of months ahead to predict")
    parser.add_argument('--model-type', choices=['linear', 'random_forest', 'both'], default='both',
                       help="Type of model to train")
    
    args = parser.parse_args()
    
    # Update file path if specified
    CoffeePriceLoader.file_path = args.coffee_file
    
    main() 