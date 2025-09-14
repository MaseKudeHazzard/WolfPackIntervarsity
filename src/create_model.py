import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset
n_samples = 1000
data = {
    'transaction_frequency': np.random.uniform(0, 20, n_samples),  # Transactions per month (0-20)
    'avg_transaction_amount': np.random.uniform(10, 200, n_samples),  # Average amount in USD (10-200)
    'utility_payment_consistency': np.random.uniform(0, 1, n_samples),  # Consistency score (0-1)
    'airtime_topup_frequency': np.random.uniform(0, 10, n_samples),  # Top-ups per month (0-10)
}

# Create DataFrame
df = pd.DataFrame(data)

# Create synthetic target variable
df['score'] = (
    0.3 * df['transaction_frequency'] +
    0.2 * df['avg_transaction_amount'] +
    0.4 * df['utility_payment_consistency'] * 10 +
    0.1 * df['airtime_topup_frequency']
)
df['target'] = (df['score'] > df['score'].median()).astype(int)

# Features and target
X = df[['transaction_frequency', 'avg_transaction_amount', 'utility_payment_consistency', 'airtime_topup_frequency']]
y = df['target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_scaled, y)

# Save the model, scaler, and a background dataset for SHAP
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
# Save a subset of scaled data (100 samples) for SHAP
background_data = X_scaled[:100]
joblib.dump(background_data, 'background_data.pkl')

print("Model, scaler, and background data saved as 'logistic_regression_model.pkl', 'scaler.pkl', and 'background_data.pkl'")
