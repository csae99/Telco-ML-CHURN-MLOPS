import os
import pandas as pd
import joblib  # Instead of mlflow

# === MODEL LOADING CONFIGURATION ===
MODEL_DIR = "/app/model"

# Load the trained model and preprocessor from .pkl files
try:
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessing.pkl"))
    print(f"✅ Model and preprocessor loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"❌ Failed to load model or preprocessor from {MODEL_DIR}: {e}")
    raise

# === FEATURE SCHEMA LOADING ===
# Same as before
try:
    feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
    with open(feature_file) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

# === FEATURE TRANSFORMATION CONSTANTS ===
# (No change here)
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    # Keep your feature transformation code as it is
    # It ensures consistency with the training preprocessing pipeline
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    # Type coercion for numeric columns
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(0)
    
    # Apply binary encoding
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map(mapping)
                .astype("Int64")
                .fillna(0)
                .astype(int)
            )
    
    # One-hot encode remaining categorical features
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    
    # Convert boolean to integer
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:  # Or you can use `if not bool_cols.empty:`
        df[bool_cols] = df[bool_cols].astype(int)
    print(f"Boolean columns found: {bool_cols}")
    # Ensure feature alignment
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return df

def predict(input_dict: dict) -> str:
    # Convert input to DataFrame
    df = pd.DataFrame([input_dict])
    
    # Apply feature transformations
    df_enc = _serve_transform(df)
    
    # Model prediction
    try:
        preds = model.predict(df_enc)
        
        if hasattr(preds, "tolist"):
            preds = preds.tolist()  # Convert numpy array to list
        
        # Get the prediction
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
    
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    # Convert prediction to business-friendly output
    if result == 1:
        return "Likely to churn"
    else:
        return "Not likely to churn"