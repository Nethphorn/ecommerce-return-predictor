# E-Commerce Return Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-orange)
![Streamlit](https://img.shields.io/badge/Web%20App-Streamlit-red)

## Project Overview

This project is an end-to-end Machine Learning pipeline that predicts whether a customer is likely to return an e-commerce purchase. As part of a class project, I start by analyzing customer demographics and purchase details, businesses can identify high-risk transactions and optimize their shipping and return policies.

### Features

- **Custom Data Generation:** Simulates realistic e-commerce purchase histories including customer age, product categories, prices, and return statuses.
- **Predictive Modeling:** Uses a `RandomForestClassifier` trained on the synthetic dataset to predict return probabilities.
- **Interactive Web App:** A clean, user-friendly Streamlit interface allowing users to input purchase details and get instant predictions.

---

## How to Run Locally

### 1. Install Dependencies

Make sure you have Python installed, then install the required packages:

```bash
pip install -r requirements.txt
```

_(If `streamlit` or `joblib` are not in your requirements.txt, you can install them directly with `pip install pandas scikit-learn streamlit joblib`)_

### 2. Generate Data & Train the Model

Run the data generation script to create the dataset, then train and save the model:

```bash
python src/generate_data.py
python src/train_model.py
```

_Note: This will create a `data/raw_orders.csv` file and save the trained model into a `models/` directory._

### 3. Launch the Web App

Start the Streamlit application:

```bash
streamlit run src/app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`.

---

## How the Model Works

1. **Features:** The model makes predictions based on:
   - `Age`: The customer's age.
   - `Category`: The type of product purchased (Electronics, Clothing, Home, Books, Beauty).
   - `Amount`: The price paid for the item.
2. **Preprocessing:** Categorical variables (like 'Category') are One-Hot Encoded so the machine learning model can understand them.
3. **Algorithm:** A Random Forest Classifier analyzes the patterns to determine the likelihood of a return.

---

## Repository Structure

```text
learning_ai/
│
├── data/                   # Generated synthetic datasets
├── models/                 # Saved machine learning models (.pkl)
├── src/                    # Source code
│   ├── generate_data.py    # Script to create synthetic e-commerce data
│   ├── analyze_data.py     # Script for basic data exploration
│   ├── train_model.py      # Script to preprocess data and train the AI model
│   └── app.py              # Streamlit web application
│
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
```

---

_Created as part of a Machine Learning portfolio._
