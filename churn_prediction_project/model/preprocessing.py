import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path):
    df = pd.read_csv(path)

    print(df.columns)  # Debug

    # Fix TotalCharges 
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()

    # Target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Drop ID column
    df = df.drop('customerID', axis=1)

    # One-hot encoding for categorical columns
    df = pd.get_dummies(df, drop_first=True)

    # Split X and y
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y