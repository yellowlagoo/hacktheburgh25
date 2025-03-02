import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import csv 
from io import StringIO

def clean_csv(file_path):
    print(f"Cleaning file: {file_path}")
    
    # Read the file line by line
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    print(f"Total lines in file: {len(lines)}")
    
    # Clean the data: remove title/header and ensure proper format
    clean_lines = []
    header_added = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue          
        # Skip title line containing "Coffee Prices - 45 Year Historical Chart"
        if "coffee prices" in line.lower() or "historical chart" in line.lower():
            print(f"Skipping title line {i+1}: {line}")
            continue
            
        # Process the header line
        if "date" in line.lower() and "value" in line.lower() or "data" in line.lower():
            if not header_added:
                clean_lines.append("date,value")  # Add standardized header
                header_added = True
                print(f"Added standardized header")
            continue
            
        # Keep only lines that look like dates (start with digit and contain dash)
        if not (line[0].isdigit() and '-' in line[:10]):
            print(f"Skipping non-date line {i+1}: {line}")
            continue
            
        # Process data lines
        if ',' in line:
            parts = line.split(',')
            if len(parts) >= 2:
                date_part = parts[0].strip()
                value_part = parts[1].strip()
                clean_lines.append(f"{date_part},{value_part}")
    
    # Create the cleaned data string
    clean_data = '\n'.join(clean_lines)
    print(f"\nCleaned data has {len(clean_lines)} lines")
    
    return clean_data

# Load the datasets
coffee_df = pd.read_csv("new_modelling/coffee-prices-historical-chart-data.csv", skiprows=15)
sentiment_df = pd.read_csv("new_modelling/UMCSENT.csv")

print("Coffee data columns:", coffee_df.columns.tolist())
print("Sentiment data columns:", sentiment_df.columns.tolist())

# Rename columns to make them more descriptive
coffee_df.rename(columns={'date': 'Date', 'value': 'Coffee Price'}, inplace=True)
sentiment_df.rename(columns={'observation_date': 'Date', 'UMCSENT': 'Sentiment Score'}, inplace=True)
clean_coffee = clean_csv('coffee-prices-historical-chart-data.csv')
coffee_df = pd.read_csv(StringIO(clean_coffee))
sentiment_df = pd.read_csv("UMCSENT.csv")

# Convert date columns to datetime format
coffee_df['date'] = pd.to_datetime(coffee_df['date'])
sentiment_df['observation_date'] = pd.to_datetime(sentiment_df['observation_date'])

# Aggregate coffee prices to monthly averages
coffee_df.set_index('date', inplace=True)
monthly_coffee = coffee_df.resample('M', on='date').mean().reset_index()
monthly_coffee = monthly_coffee.rename(columns={'date': 'observation_date'})

# Merge with sentiment data on the date column
merged_df = pd.merge(monthly_coffee, sentiment_df, on="Date", how="inner")
print(f"Shape of merged data: {merged_df.shape}")
print("First few rows of merged data:")
print(merged_df.head())
merged_df = pd.merge(monthly_coffee, sentiment_df, on="observation_date", how="inner")

# Define features (X) and target (y)
x = merged_df[['Coffee Price']]  # Feature
y = merged_df['Sentiment Score']  # Target


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
#y_pred = model.predict(x)

# Evaluate the model
#r2 = r2_score(y, y_pred)
#mae = mean_absolute_error(y, y_pred)

#print(f"\nModel R² Score: {r2:.4f}")
#print(f"Mean Absolute Error: {mae:.4f}")

# Plot Actual vs Predicted
#plt.figure(figsize=(10, 6))
#plt.scatter(y, y_pred)
#plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
#plt.xlabel("Actual Sentiment Score")
#plt.ylabel("Predicted Sentiment Score")
#plt.title("Coffee Price → Market Sentiment Prediction")
#plt.savefig("sentiment_prediction.png")
#print("Plot saved as sentiment_prediction.png")
