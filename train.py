import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0)

df = pd.DataFrame(X, columns=["X1", "X2", "X3", "X4"])

# Train a sample model
model = RandomForestClassifier(n_estimators=100)
model.fit(df, y)

# Save the model
joblib.dump(model, 'model.pkl')
