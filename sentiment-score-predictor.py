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

# Replace your current CSV reading code with this:
try:
    # Load and clean the CSV data
    clean_data = clean_csv('coffee-prices-historical-chart-data.csv')
    
    # Create DataFrame from cleaned data
    from io import StringIO
    coffee_df = pd.read_csv(StringIO(clean_data))
    
    # Convert the date column to datetime with error handling
    coffee_df['date'] = pd.to_datetime(coffee_df['date'], errors='coerce')
    
    # Drop rows with invalid dates
    invalid_dates = coffee_df[coffee_df['date'].isna()]
    if len(invalid_dates) > 0:
        print(f"Dropping {len(invalid_dates)} rows with invalid dates")
        coffee_df = coffee_df.dropna(subset=['date'])
    
    # Make sure values are numeric
    coffee_df['value'] = pd.to_numeric(coffee_df['value'], errors='coerce')
    coffee_df = coffee_df.dropna(subset=['value'])
    
    print("\nCleaned DataFrame shape:", coffee_df.shape)
    print(coffee_df.head())
    
    # Continue with your existing linear regression code...
    # ...
    
except Exception as e:
    print(f"\nError: {str(e)}")
    import traceback
    traceback.print_exc()