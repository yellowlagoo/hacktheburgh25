import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('weather_data_collector')

class WeatherDataGenerator:
    """
    Class to generate realistic synthetic weather data for coffee-producing countries
    
    Note: In a production environment, this would be replaced with actual API calls
    to weather data providers like OpenWeatherMap, WeatherAPI, or NOAA.
    """
    
    def __init__(self):
        # Define climate characteristics for each country
        self.climate_profiles = {
            'Brazil': {
                'rainfall': {
                    'base': 100,  # Base rainfall in mm
                    'variance': 40,  # Month-to-month variance
                    'seasonal_amplitude': 80,  # Seasonal change amplitude
                    'peak_month': 1,  # January (rainy season)
                },
                'sunshine': {
                    'base': 180,  # Base sunshine hours per month
                    'variance': 20,
                    'seasonal_amplitude': 40,
                    'peak_month': 7,  # July (dry season has more sunshine)
                }
            },
            'Vietnam': {
                'rainfall': {
                    'base': 80,
                    'variance': 30,
                    'seasonal_amplitude': 120,
                    'peak_month': 8,  # August (monsoon season)
                },
                'sunshine': {
                    'base': 200,
                    'variance': 25,
                    'seasonal_amplitude': 60,
                    'peak_month': 2,  # February (dry season)
                }
            },
            'Colombia': {
                'rainfall': {
                    'base': 150,
                    'variance': 35,
                    'seasonal_amplitude': 60,
                    'peak_month': 10,  # October (rainy season)
                },
                'sunshine': {
                    'base': 160,
                    'variance': 15,
                    'seasonal_amplitude': 30,
                    'peak_month': 1,  # January (more sunshine)
                }
            }
        }
    
    def generate_monthly_data(self, start_date, end_date):
        """
        Generate monthly weather data for all countries
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with monthly weather data
        """
        # Convert dates to datetime objects
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Generate monthly date range (first day of each month)
        date_range = pd.date_range(
            start=start.replace(day=1),
            end=end.replace(day=1),
            freq='MS'  # Month start frequency
        )
        
        # Create empty list to store data
        weather_data = []
        
        # Generate data for each country
        for country in self.climate_profiles:
            profile = self.climate_profiles[country]
            
            # Set random seed for reproducibility but different for each country
            random.seed(hash(country) % 10000)
            np.random.seed(hash(country) % 10000)
            
            for date in date_range:
                month = date.month
                year = date.year
                
                # Calculate seasonal factors (sine wave pattern throughout the year)
                rainfall_season = self._seasonal_factor(
                    month, 
                    profile['rainfall']['peak_month'], 
                    profile['rainfall']['seasonal_amplitude']
                )
                
                sunshine_season = self._seasonal_factor(
                    month, 
                    profile['sunshine']['peak_month'], 
                    profile['sunshine']['seasonal_amplitude']
                )
                
                # Add yearly climate variations (some years wetter/drier)
                yearly_factor = self._yearly_factor(year)
                
                # Calculate final values with random variations
                rainfall = (
                    profile['rainfall']['base'] + 
                    rainfall_season + 
                    yearly_factor * 20 + 
                    np.random.normal(0, profile['rainfall']['variance'])
                )
                
                sunshine = (
                    profile['sunshine']['base'] + 
                    sunshine_season + 
                    yearly_factor * -10 +  # Negative correlation with rainfall
                    np.random.normal(0, profile['sunshine']['variance'])
                )
                
                # Ensure values are reasonable (no negative values)
                rainfall = max(0, rainfall)
                sunshine = max(0, min(sunshine, 365))  # Max sunshine hours in a month
                
                # Format date as 01-MM-YYYY
                formatted_date = date.strftime('01-%m-%Y')
                
                # Add to data list
                weather_data.append({
                    'Date': formatted_date,
                    'Avg Rainfall (mm)': round(rainfall, 1),
                    'Avg Sunshine (hours)': round(sunshine, 1),
                    'Country': country
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(weather_data)
        
        return df
    
    def _seasonal_factor(self, month, peak_month, amplitude):
        """Calculate seasonal factor using sine wave"""
        # Convert month difference to radians (2π = full year)
        month_diff = (month - peak_month) % 12
        if month_diff > 6:
            month_diff = 12 - month_diff
        
        # Calculate as cosine wave (1 at peak, -1 at opposite)
        # Convert to 0-2π range
        angle = month_diff * (2 * np.pi / 12)
        factor = np.cos(angle)
        
        return factor * amplitude
    
    def _yearly_factor(self, year):
        """Generate yearly climate variations (El Niño/La Niña effects)"""
        # Use a sine wave with ~3-7 year periodicity (El Niño-like cycles)
        base_cycle = np.sin(2 * np.pi * (year - 2010) / 5)
        
        # Add some random variation
        factor = base_cycle + np.random.normal(0, 0.2)
        
        # Constrain between -1 and 1
        return max(-1, min(1, factor))


def fetch_weather_data(start_date, end_date, output_file='weather_data.csv'):
    """
    Main function to fetch, process and save weather data
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_file: Path to save the CSV file
    """
    logger.info(f"Fetching weather data from {start_date} to {end_date}")
    
    # In a real application, this would call an actual weather API
    # For this example, we'll generate synthetic data
    generator = WeatherDataGenerator()
    
    # Generate weather data
    weather_data = generator.generate_monthly_data(start_date, end_date)
    
    # Verify no missing values
    missing_values = weather_data.isnull().sum().sum()
    if missing_values > 0:
        logger.warning(f"Found {missing_values} missing values. Interpolating...")
        weather_data = weather_data.interpolate()
    
    # Save to CSV
    output_path = Path(output_file)
    weather_data.to_csv(output_path, index=False)
    
    logger.info(f"Weather data saved to {os.path.abspath(output_path)}")
    print(f"Weather data successfully generated and saved to: {os.path.abspath(output_path)}")
    
    # Display a sample of the data
    print("\nSample of the weather data:")
    print(weather_data.head(9))  # Show 3 months for each country
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate historical weather data for coffee-producing countries")
    parser.add_argument('--start', type=str, default='2018-01-01', 
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end', type=str, default='2023-12-31', 
                        help='End date in YYYY-MM-DD format')
    parser.add_argument('--output', type=str, default='weather_data.csv', 
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Fetch and save weather data
    fetch_weather_data(args.start, args.end, args.output)