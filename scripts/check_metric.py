import mlflow
import pandas as pd
import json
import sys
import os
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

# ================================
# CONFIG
# ================================
experiment_name = "Telco Churn"
mlflow.set_tracking_uri("http://mlflow:5000")

# ================================
# LOAD LATEST RUN
# ================================
experiment = mlflow.get_experiment_by_name(experiment_name)
runs = mlflow.search_runs(experiment_ids=experiment.experiment_id)

# sort manually (MLflow doesn't support ORDER BY)
runs = runs.sort_values(by="start_time", ascending=False)
latest_run = runs.iloc[0]

run_id = latest_run.run_id
print(f"🔍 Using run_id: {run_id}")

# ================================
# LOAD MODEL
# ================================
model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

# ================================
# LOAD FEATURE COLUMNS (FIXED)
# ================================
feature_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="feature_columns.txt"
)

with open(feature_path) as f:
    feature_columns = [line.strip() for line in f if line.strip()]

print(f"✅ Loaded {len(feature_columns)} features")

# ================================
# LOAD TEST DATA
# ================================
test_data_path = "data/rawS/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df_test = pd.read_csv(test_data_path)

target_column = "Churn"

# ================================
# APPLY SAME PIPELINE
# ================================
df_test = preprocess_data(df_test)
df_test_enc = build_features(df_test, target_col=target_column)

X_test = df_test_enc.drop(columns=[target_column])
y_test = df_test_enc[target_column]

# CRITICAL: align columns
X_test = X_test.reindex(columns=feature_columns, fill_value=0)

# ================================
# PREDICT
# ================================
y_proba = model.predict_proba(X_test)[:, 1]

# Load threshold
try:
    with open("best_params.json") as f:
        threshold = json.load(f).get("threshold", 0.35)
except:
    threshold = 0.35

y_pred = (y_proba >= threshold).astype(int)

# ================================
# METRICS
# ================================
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n📊 Model Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")

# ================================
# FAIL CONDITIONS (CI/CD GATE)
# ================================
if recall < 0.85:
    print("❌ Recall too low")
    sys.exit(1)

if precision < 0.45:
    print("❌ Precision too low")
    sys.exit(1)

if f1 < 0.58:
    print("❌ F1 too low")
    sys.exit(1)

if roc_auc < 0.83:
    print("❌ ROC AUC too low")
    sys.exit(1)

print("✅ Model passed quality checks!")
sys.exit(0)