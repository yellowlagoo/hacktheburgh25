import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load the datasets
coffee_df = pd.read_csv("/mnt/data/coffee-prices-historical-chart-data.csv")
sentiment_df = pd.read_csv("/mnt/data/UMCSENT.csv")

# Convert date columns to datetime format
coffee_df['Date'] = pd.to_datetime(coffee_df['Date'])
sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])

# Aggregate coffee prices to monthly averages
coffee_df.set_index('Date', inplace=True)
monthly_coffee = coffee_df.resample('M').mean().reset_index()

# Merge with sentiment data on the date column
merged_df = pd.merge(monthly_coffee, sentiment_df, on="Date", how="inner")

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

# Make predictions
#y_pred = model.predict()

# Evaluate the model
#r2 = r2_score(y_test, y_pred)
#mae = mean_absolute_error(y_test, y_pred)

#print(f"Model R² Score: {r2:.4f}")
#print(f"Mean Absolute Error: {mae:.4f}")

# Plot Actual vs Predicted
#plt.scatter(y_test, y_pred)
#plt.xlabel("Actual Sentiment Score")
#plt.ylabel("Predicted Sentiment Score")
#plt.title("Coffee Price → Market Sentiment Prediction")
#plt.show()
