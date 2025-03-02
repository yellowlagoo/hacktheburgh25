import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import csv
from io import StringIO

# Step 1: Manually read and clean the problematic CSV file
def clean_csv(file_path):
    print(f"Cleaning file: {file_path}")
    
    # Read the file line by line
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    print(f"Total lines in file: {len(lines)}")
    
    # Examine the problematic area
    print("\nExamining lines around line 16:")
    for i in range(max(0, 15-2), min(len(lines), 15+3)):
        print(f"Line {i+1}: {lines[i].strip()}")
    
    # Clean the data: remove the problematic lines and ensure proper format
    clean_lines = []
    header_seen = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Handle the header (first non-empty line)
        if not header_seen:
            clean_lines.append("date,value")  # Force a clean header
            header_seen = True
            continue
        
        # Skip the problematic line 16 and any other "date, value" repeats
        if "date" in line.lower() and "value" in line.lower():
            print(f"Skipping header-like line {i+1}: {line}")
            continue
            
        # Clean data lines: ensure single field if that's what's expected
        # This handles the case where the file might actually be single-column
        if ',' in line:
            parts = line.split(',')
            # If there are multiple parts, keep only what we need
            if len(parts) > 1:
                # Keep the date and first value
                date_part = parts[0].strip()
                value_part = parts[1].strip()
                clean_lines.append(f"{date_part},{value_part}")
            else:
                clean_lines.append(line)
        else:
            # If no comma, this might be just a date - add a placeholder value
            clean_lines.append(f"{line},0")
    
    # Join the cleaned lines
    clean_data = '\n'.join(clean_lines)
    print(f"\nCleaned data has {len(clean_lines)} lines")
    
    return clean_data

# Step 2: Process the data for Prophet
def prepare_prophet_dataframe(clean_data):
    # Read the cleaned data
    df = pd.read_csv(StringIO(clean_data))
    
    print("\nDataFrame created from cleaned data:")
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("\nSample data:")
    print(df.head())
    
    # Rename columns for Prophet
    prophet_df = pd.DataFrame()
    prophet_df['ds'] = pd.to_datetime(df['date'], errors='coerce')
    prophet_df['y'] = pd.to_numeric(df['value'], errors='coerce')
    
    # Drop rows with missing or invalid values
    prophet_df = prophet_df.dropna()
    
    print(f"\nProphet DataFrame created with {len(prophet_df)} valid rows")
    return prophet_df

# Step 3: Run the Prophet model
def run_prophet_forecast(prophet_df):
    # Initialize and fit the model
    m = Prophet()
    m.fit(prophet_df)
    
    # Create future dataframe for prediction (next 365 days)
    future = m.make_future_dataframe(periods=365)
    
    # Make predictions
    forecast = m.predict(future)
    print(forecast)
    
    # Plot the forecast
    fig = m.plot(forecast)
    plt.title('Coffee Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.savefig('coffee_forecast.png')
    
    # Plot forecast components
    fig2 = m.plot_components(forecast)
    plt.savefig('coffee_components.png')
    
    print("\nForecast for the next 10 periods:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))
    
    print("\nForecast plots saved as 'coffee_forecast.png' and 'coffee_components.png'")
    return forecast

# Main execution
try:
    # Step 1: Clean the CSV file
    clean_data = clean_csv('coffee-prices-historical-chart-data.csv')
    
    # Step 2: Prepare the data for Prophet
    prophet_df = prepare_prophet_dataframe(clean_data)
    
    # Step 3: Run the forecast
    forecast = run_prophet_forecast(prophet_df)
    
except Exception as e:
    print(f"\nError: {str(e)}")
    import traceback
    traceback.print_exc()

