import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("1. Loading the data...")
df = pd.read_csv('data/raw_orders.csv')

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING (Preparing data for the AI)
# ---------------------------------------------------------
print("2. Preprocessing data...")

# We only want to use columns that help predict a return
# 'user_id' and 'order_date' don't help the AI, so we drop them.
features = df[['age', 'category', 'amount']]

# The target is what we are trying to predict
target = df['is_return']

# Machine Learning models need numbers, not words!
# pd.get_dummies converts our 'category' text into 1s and 0s (One-Hot Encoding)
X = pd.get_dummies(features, columns=['category'])
y = target

# ---------------------------------------------------------
# 3. SPLIT DATA INTO TRAINING AND TESTING SETS
# ---------------------------------------------------------
# We use 80% of data to teach the model, and keep 20% to test it like an exam.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 4. TRAINING THE MODEL
# ---------------------------------------------------------
print("3. Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. EVALUATING THE MODEL
# ---------------------------------------------------------
print("4. Evaluating model performance...")
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\n--- Detailed Report ---")
print(classification_report(y_test, predictions))

# ---------------------------------------------------------
# 6. SAVE THE MODEL FOR LATER (For the Web UI!)
# ---------------------------------------------------------
print("5. Saving the model...")
os.makedirs('models', exist_ok=True) # Create a "models" folder if it doesn't exist

# Save the trained model and the feature names so we can load them in our app.py later
joblib.dump(model, 'models/return_predictor_model.pkl')

# Save the exact column names the model was trained on
joblib.dump(list(X.columns), 'models/model_features.pkl')

print("Success! Model saved in the 'models' folder.")
