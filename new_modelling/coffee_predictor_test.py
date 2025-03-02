import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import csv
from io import StringIO

def clean_csv(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    clean_lines = []
    header_seen = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        if not line:
            continue
            
        if not header_seen:
            clean_lines.append("date,value")  
            header_seen = True
            continue
        
        if "date" in line.lower() and "value" in line.lower():
            print(f"Skipping header-like line {i+1}: {line}")
            continue
            
        if ',' in line:
            parts = line.split(',')
            if len(parts) > 1:
                date_part = parts[0].strip()
                value_part = parts[1].strip()
                clean_lines.append(f"{date_part},{value_part}")
            else:
                clean_lines.append(line)
        else:
            clean_lines.append(f"{line},0")
    
    clean_data = '\n'.join(clean_lines)
    print(f"\nCleaned data has {len(clean_lines)} lines")
    
    return clean_data

def prepare_prophet_dataframe(clean_data):
    df = pd.read_csv(StringIO(clean_data))
    
    print("\nDataFrame created from cleaned data:")
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("\nSample data:")
    print(df.head())
    
    prophet_df = pd.DataFrame()
    prophet_df['ds'] = pd.to_datetime(df['date'], errors='coerce')
    prophet_df['y'] = pd.to_numeric(df['value'], errors='coerce')
    
    prophet_df = prophet_df.dropna()
    
    return prophet_df

def run_prophet_forecast(prophet_df):
    m = Prophet()
    m.fit(prophet_df)
    
    future = m.make_future_dataframe(periods=365)
    
    forecast = m.predict(future)
    print(forecast)
    
    fig = m.plot(forecast)
    plt.title('Coffee Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.savefig('coffee_forecast.png')
    
    fig2 = m.plot_components(forecast)
    plt.savefig('coffee_components.png')
    
    print("\nForecast for the next 10 periods:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))
    
    print("\nForecast plots saved as 'coffee_forecast.png' and 'coffee_components.png'")
    return forecast

clean_data = clean_csv('coffee-prices-historical-chart-data.csv')
prophet_df = prepare_prophet_dataframe(clean_data)
forecast = run_prophet_forecast(prophet_df)
    

