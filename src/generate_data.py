import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_ecommerce_data(n_rows=1000):
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_rows)]
    
    # Generate customer data
    user_ids = np.random.randint(1000, 2000, n_rows)
    ages = np.random.randint(18, 70, n_rows)
    
    # Categories and prices
    categories = ['Electronics', 'Clothing', 'Home', 'Books', 'Beauty']
    cat_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
    
    selected_cats = np.random.choice(categories, n_rows, p=cat_weights)
    
    # Price logic based on category
    base_prices = {'Electronics': 500, 'Clothing': 50, 'Home': 150, 'Books': 25, 'Beauty': 40}
    amounts = [base_prices[cat] + np.random.normal(0, base_prices[cat]*0.2) for cat in selected_cats]
    
    # Return logic (e.g., higher price items or specific categories might have higher return rates)
    return_probs = [0.1 + (amt / 1000) * 0.2 for amt in amounts]
    is_return = [np.random.choice([0, 1], p=[1-p, p]) for p in return_probs]
    
    df = pd.DataFrame({
        'order_date': dates,
        'user_id': user_ids,
        'age': ages,
        'category': selected_cats,
        'amount': amounts,
        'is_return': is_return
    })
    
    # Sort by date for realism
    df = df.sort_values('order_date').reset_index(drop=True)
    
    # Save to data directory
    os.makedirs('data', exist_ok=True)
    file_path = 'data/raw_orders.csv'
    df.to_csv(file_path, index=False)
    print(f"Dataset generated with {n_rows} rows at {file_path}")
    return df

if __name__ == "__main__":
    generate_ecommerce_data()
