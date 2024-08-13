import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
import pickle

data = pd.read_csv("Cleaned_data.csv")

categorical_features = ["location"]
numeric_features = ["bhk", "bath", "total_sqft"]
X = data[categorical_features + numeric_features]
y = data["price"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(), categorical_features),
    ]
)


pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", Ridge())])
pipe.fit(X, y)


with open("RidgeModel.pkl", "wb") as f:
    pickle.dump(pipe, f)

print("Model training complete and saved as 'RidgeModel.pkl'.")
