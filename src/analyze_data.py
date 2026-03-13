import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv('data/raw_orders.csv')

# 2. Basic Stats (The 'Health Check')
print("--- Data Overview ---")
print(df.info()) # Shows column types and if any data is missing
print("\n--- Summary Statistics ---")
print(df.describe()) # Shows Mean, Min, Max for numerical columns

# 3. Business Analysis: Sales by Category
category_perf = df.groupby('category').agg({
    'amount': 'sum',
    'is_return': 'mean' # This gives us the return percentage!
}).sort_values(by='amount', ascending=False)

print("\n--- Category Performance ---")
print(category_perf)

# 4. Visualization: Price vs Returns
# We want to see if our "secret rule" (expensive items = more returns) is visible
plt.figure(figsize=(10, 6))
plt.scatter(df['amount'], df['is_return'], alpha=0.1)
plt.title('Order Amount vs Return Status')
plt.xlabel('Amount ($)')
plt.ylabel('Returned (1 = Yes, 0 = No)')
plt.savefig('data/price_vs_returns.png')
print("\nVisualization saved to data/price_vs_returns.png")
