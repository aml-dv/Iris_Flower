import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
iris_data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['species'])

# Encode species column
iris_data['species'] = iris_data['species'].astype(int)

# Split dataset
X = iris_data.drop(columns=['species'], axis=1)
Y = iris_data['species']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Train models
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)

# Save models
with open('model.pkl', 'wb') as f:
    pickle.dump(rfc, f)

print("Model training complete. Model saved as 'model.pkl'.")