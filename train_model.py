import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("dataset/crop_production.csv")

print(" Dataset loaded successfully!")
print("Total rows:", len(data))
print("Columns:", data.columns.tolist())


# ==========================================
# 2. SELECT FEATURES AND TARGET
# ==========================================

# User will enter these values in our live UI
features = ["State", "Crop", "Rainfall"]

X = data[features]
y = data["Yield"]


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# 4. PREPROCESSING
# ==========================================

categorical_features = ["State", "Crop"]
numeric_features = ["Rainfall"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# ==========================================
# 5. RANDOM FOREST MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 6. CREATE PIPELINE
# ==========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

print("\n Training model...")

pipeline.fit(X_train, y_train)

print(" Model training completed!")


# ==========================================
# 8. MODEL EVALUATION
# ==========================================

predictions = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\n========== MODEL PERFORMANCE ==========")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")
print("=======================================")


# ==========================================
# 9. SAVE MODEL USING JOBLIB
# ==========================================

model_path = "models/crop_yield_model.pkl"

joblib.dump(pipeline, model_path)

print(f"\n Model saved successfully!")
print(f" Location: {model_path}")