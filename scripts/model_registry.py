import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier  # <-- import this!
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Example: load your dataset
df = pd.read_csv("/home/jovyan/work/Telco-Customer-Churn-ML/data/processed/WA_Fn-UseC_-Telco-Customer-Churn.csv")
target = "Churn"
X = df.drop(columns=[target])
y = df[target]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow experiment
experiment_name = "Telco Churn"
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment(experiment_name)

with mlflow.start_run() as run:
    # Initialize and train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")

    # Log metrics and model
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    # Register the model
    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri, "ChurnPredictionModel")
    print(f"✅ Model registered: {model_uri}")

    # Transition model to Staging/Production
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(name="ChurnPredictionModel", version=1, stage="Staging")