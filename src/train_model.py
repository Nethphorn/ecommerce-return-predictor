import pandas as pd
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('data/raw_orders.csv')

features = df[['age', 'category', 'amount']]
target = df['is_return']

X_train, X_text, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_text)

accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2%}")
print(classification_report(y_test, predictions))

# Save the model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/return_predictor_model.pkl')
joblib.dump(model_features, 'models/model_features.pkl')

print("Success! Model saved in the 'models' folder.")
