
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Generate dummy training data
np.random.seed(42)
size = 200

X = pd.DataFrame({
    'amount_diff': np.random.uniform(0, 10, size),
    'date_diff': np.random.randint(0, 5, size),
    'dc_mirror': np.random.choice([0, 1], size),
    'ref_match': np.random.choice([0, 1], size),
    'desc_match': np.random.choice([0, 1], size)
})

# Simulate realistic match logic
y = ((X['amount_diff'] < 1) & (X['date_diff'] <= 1) & (X['dc_mirror'] == 1)).astype(int)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save to file
joblib.dump(model, 'recon_rf_model.pkl')
print("✅ Model saved as recon_rf_model.pkl")
