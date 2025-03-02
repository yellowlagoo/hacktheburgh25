import pandas as pd
import io
import sys

def clean_and_load_csv(file_path):
    print(f"Loading and cleaning file: {file_path}")
    
    try:
        # First attempt: try to read directly with pandas
        print("Attempting direct read with pandas...")
        df = pd.read_csv(file_path)
        print(f"Success! Columns found: {df.columns.tolist()}")
    except Exception as e:
        print(f"Direct read failed: {str(e)}")
        print("Switching to manual cleaning...")
        
        # Manual cleaning approach
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        print(f"File has {len(lines)} lines")
        
        # Show first few lines for debugging
        print("\nFirst 10 lines of the file:")
        for i in range(min(10, len(lines))):
            print(f"Line {i+1}: {lines[i].strip()}")
        
        # Create clean data with explicit header
        clean_lines = ["date,value"]  # Start with explicit header
        
        # Process data lines
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip header and descriptive lines
            if any(skip in line.lower() for skip in ["coffee prices", "historical", "chart", 
                                                    "for informational", "macrotrends"]):
                continue
                
            # Skip lines with headers
            if "date" in line.lower() and "value" in line.lower():
                continue
                
            # Only keep lines that look like data (start with year)
            if line[0].isdigit() and '-' in line[:10]:
                # Split by comma and keep date and value
                parts = line.split(',')
                if len(parts) >= 2:
                    date_part = parts[0].strip()
                    value_part = parts[1].strip()
                    clean_lines.append(f"{date_part},{value_part}")
        
        print(f"Created {len(clean_lines)} clean lines")
        
        # Create DataFrame from cleaned data
        data_str = '\n'.join(clean_lines)
        df = pd.read_csv(io.StringIO(data_str))
        
    # Show DataFrame info
    print("\nDataFrame info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    return df

# Load the coffee price data
try:
    # Load and clean the coffee price data
    coffee_df = clean_and_load_csv('coffee-prices-historical-chart-data.csv')
    
    # Ensure we have the expected columns, rename if needed
    if 'date' not in coffee_df.columns:
        # If we have exactly two columns, assume first is date, second is value
        if len(coffee_df.columns) == 2:
            print(f"Renaming columns from {coffee_df.columns.tolist()} to ['date', 'value']")
            coffee_df.columns = ['date', 'value']
        else:
            # Look for columns that might contain dates
            for col in coffee_df.columns:
                sample = coffee_df[col].astype(str).iloc[:5]
                if all('-' in str(x) for x in sample):
                    print(f"Column '{col}' looks like dates. Renaming to 'date'")
                    coffee_df = coffee_df.rename(columns={col: 'date'})
                    # Find a numeric column for 'value'
                    for val_col in coffee_df.columns:
                        if val_col != 'date':
                            try:
                                # Check if column can be converted to numeric
                                pd.to_numeric(coffee_df[val_col])
                                coffee_df = coffee_df.rename(columns={val_col: 'value'})
                                print(f"Column '{val_col}' looks numeric. Renaming to 'value'")
                                break
                            except:
                                continue
                    break
    
    # Convert date to datetime
    print("\nConverting date column to datetime...")
    coffee_df['date'] = pd.to_datetime(coffee_df['date'], errors='coerce')
    
    # Drop rows with invalid dates
    invalid_dates = coffee_df['date'].isna().sum()
    if invalid_dates > 0:
        print(f"Dropping {invalid_dates} rows with invalid dates")
        coffee_df = coffee_df.dropna(subset=['date'])
    
    # Ensure value column is numeric
    coffee_df['value'] = pd.to_numeric(coffee_df['value'], errors='coerce')
    
    # For debugging, show the final DataFrame
    print("\nFinal coffee_df:")
    print(f"Shape: {coffee_df.shape}")
    print(f"Columns: {coffee_df.columns.tolist()}")
    print(coffee_df.head())
    
    # Now resample to monthly - using 'ME' to avoid deprecation warning
    print("\nResampling to monthly data...")
    monthly_coffee = coffee_df.resample('ME', on='date').mean().reset_index()
    
    print("\nMonthly coffee data:")
    print(f"Shape: {monthly_coffee.shape}")
    print(f"Columns: {monthly_coffee.columns.tolist()}")
    print(monthly_coffee.head())
    
    # Rename for the merge
    monthly_coffee = monthly_coffee.rename(columns={'date': 'observation_date'})
    
    print("\nAfter renaming, columns:", monthly_coffee.columns.tolist())
    
    # Continue with your existing code for merging, etc.
    # merged_df = pd.merge(monthly_coffee, sentiment_df, on="observation_date", how="inner")
    
    # For debugging, if merge fails
    # print("sentiment_df columns:", sentiment_df.columns.tolist())
    # print("sentiment_df sample data:")
    # print(sentiment_df.head())
    
except Exception as e:
    print(f"\nError: {str(e)}")
    import traceback
    traceback.print_exc()