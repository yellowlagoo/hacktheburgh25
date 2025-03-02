import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load the datasets
coffee_df = pd.read_csv("new_modelling/coffee-prices-historical-chart-data.csv", skiprows=15)
sentiment_df = pd.read_csv("new_modelling/UMCSENT.csv")

print("Coffee data columns:", coffee_df.columns.tolist())
print("Sentiment data columns:", sentiment_df.columns.tolist())

# Rename columns to make them more descriptive
coffee_df.rename(columns={'date': 'Date', 'value': 'Coffee Price'}, inplace=True)
sentiment_df.rename(columns={'observation_date': 'Date', 'UMCSENT': 'Sentiment Score'}, inplace=True)

# Convert date columns to datetime format
coffee_df['Date'] = pd.to_datetime(coffee_df['Date'])
sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])

# Aggregate coffee prices to monthly averages
coffee_df.set_index('Date', inplace=True)
monthly_coffee = coffee_df.resample('M').mean().reset_index()

# Merge with sentiment data on the date column
merged_df = pd.merge(monthly_coffee, sentiment_df, on="Date", how="inner")
print(f"Shape of merged data: {merged_df.shape}")
print("First few rows of merged data:")
print(merged_df.head())

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
