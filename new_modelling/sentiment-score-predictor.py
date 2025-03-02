import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import os

# Function to clean the coffee price CSV file
def clean_csv(file_path):
    print(f"Cleaning file: {file_path}")
    
    # Read the file line by line
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    print(f"Total lines in file: {len(lines)}")
    
    # Find the data section (skip header information)
    data_start = 0
    for i, line in enumerate(lines):
        if "date,value" in line.lower() or "data,value" in line.lower():
            data_start = i + 1
            break
    
    # If we didn't find the header, try to detect it another way
    if data_start == 0:
        for i, line in enumerate(lines):
            if i > 10 and ',' in line and line[0].isdigit():
                data_start = i
                break
    
    print(f"Data starts at line {data_start}")
    
    # Process only the data section
    clean_data = "date,value\n"  # Start with header
    for line in lines[data_start:]:
        line = line.strip()
        if line and ',' in line and line[0].isdigit():
            parts = line.split(',')
            if len(parts) >= 2:
                clean_data += f"{parts[0]},{parts[1]}\n"
    
    return clean_data

try:
    # Define file paths - try different possible locations
    coffee_paths = [
        "new_modelling/coffee-prices-historical-chart-data.csv",
        "coffee-prices-historical-chart-data.csv"
    ]
    
    sentiment_paths = [
        "new_modelling/UMCSENT.csv", 
        "UMCSENT.csv"
    ]
    
    # Find the coffee price data file
    coffee_file = None
    for path in coffee_paths:
        if os.path.exists(path):
            coffee_file = path
            break
    
    if not coffee_file:
        raise FileNotFoundError("Coffee price data file not found")
    
    # Find the sentiment data file
    sentiment_file = None
    for path in sentiment_paths:
        if os.path.exists(path):
            sentiment_file = path
            break
    
    if not sentiment_file:
        raise FileNotFoundError("Sentiment data file not found")
    
    print(f"Using coffee data from: {coffee_file}")
    print(f"Using sentiment data from: {sentiment_file}")
    
    # Clean and load the coffee price data
    from io import StringIO
    clean_coffee_data = clean_csv(coffee_file)
    coffee_df = pd.read_csv(StringIO(clean_coffee_data))
    
    # Load sentiment data
    sentiment_df = pd.read_csv(sentiment_file)
    
    print("Coffee data columns:", coffee_df.columns.tolist())
    print("Sentiment data columns:", sentiment_df.columns.tolist())
    
    # Convert date columns to datetime format
    coffee_df['date'] = pd.to_datetime(coffee_df['date'])
    sentiment_df['observation_date'] = pd.to_datetime(sentiment_df['observation_date'])
    
    # Rename the coffee price column for clarity
    coffee_df = coffee_df.rename(columns={'value': 'Coffee Price'})
    
    # Aggregate coffee prices to monthly averages to match sentiment data frequency
    coffee_df.set_index('date', inplace=True)
    monthly_coffee = coffee_df.resample('M').mean().reset_index()
    monthly_coffee = monthly_coffee.rename(columns={'date': 'observation_date'})
    
    # Merge the datasets on the date column
    merged_df = pd.merge(monthly_coffee, sentiment_df, on="observation_date", how="inner")
    
    print(f"Shape of merged data: {merged_df.shape}")
    if merged_df.shape[0] == 0:
        raise ValueError("No matching data points between coffee prices and sentiment data")
    
    print("First few rows of merged data:")
    print(merged_df.head())
    
    # Define features (X) and target (y)
    x = merged_df[['Coffee Price']]  # Feature
    y = merged_df['UMCSENT']  # Target variable (sentiment score)
    
    # Train a Linear Regression Model
    model = LinearRegression()
    model.fit(x, y)
    
    # Output the linear regression formula
    coefficient = model.coef_[0]
    intercept = model.intercept_
    formula = f"Sentiment Score = {intercept:.4f} + {coefficient:.4f} * Coffee Price"
    print("\nLinear Regression Formula:")
    print(formula)
    
    # Make predictions on the training data
    y_pred = model.predict(x)
    
    # Evaluate the model
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    print(f"\nModel R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    
    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
    plt.xlabel("Actual Sentiment Score")
    plt.ylabel("Predicted Sentiment Score")
    plt.title("Coffee Price → Market Sentiment Prediction")
    plt.savefig("sentiment_prediction.png")
    print("Plot saved as sentiment_prediction.png")
    
    # Also plot Coffee Price vs Sentiment
    plt.figure(figsize=(10, 6))
    plt.scatter(merged_df['Coffee Price'], merged_df['UMCSENT'])
    plt.plot(merged_df['Coffee Price'], y_pred, 'r-')
    plt.xlabel("Coffee Price")
    plt.ylabel("Sentiment Score")
    plt.title("Sentiment Score vs Coffee Price")
    plt.savefig("coffee_sentiment_relationship.png")
    print("Plot saved as coffee_sentiment_relationship.png")

except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
