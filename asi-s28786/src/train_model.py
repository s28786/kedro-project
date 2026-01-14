"""
train_model.py

Delivery 5: MLflow Experiment Tracking & Model Versioning
--------------------------------------------------------
This script trains four classifiers on the Iris dataset, logs all relevant metrics, parameters, and artifacts to MLflow,
selects the best model by F1-macro, saves the best model and metadata locally, and registers the best model in the MLflow Model Registry.

Requirements:
- MLflow must be installed: uv pip install mlflow
- The Iris.csv file must be present in the same directory as this script.
- Output files: app/model.joblib, app/model_meta.json

MLflow UI:
- Run: mlflow ui --backend-store-uri ./mlruns --port 5000
- Open: http://localhost:5000
"""

import os
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

EXPERIMENT_NAME = "iris-model-zoo"
REGISTERED_MODEL_NAME = "IrisModel"
VERSION = "v1.0.0"
APP_DIR = os.path.join(os.path.dirname(__file__), "app")
MODEL_PATH = os.path.join(APP_DIR, "model.joblib")
META_PATH = os.path.join(APP_DIR, "model_meta.json")
CSV_PATH = os.path.join(os.path.dirname(__file__), "Iris.csv")
MLFLOW_TRACKING_URI = os.path.join(os.path.dirname(__file__), "mlruns")

# --- Ensure output directory exists ---
os.makedirs(APP_DIR, exist_ok=True)

# --- Set MLflow tracking URI to local mlruns directory (Model Registry requires file:// URI) ---
mlflow.set_tracking_uri(f"file:///{MLFLOW_TRACKING_URI.replace(os.sep, '/')}")

# --- Load Iris data ---
df = pd.read_csv(CSV_PATH)
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Model zoo: four classifiers ---
def get_models():
    return {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=200, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier()
    }

# --- Utility: Plot and save confusion matrix ---
def plot_confusion_matrix(cm, classes, model_name, out_path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# --- MLflow logging for a single model run ---
def log_model_run(model_name, model, X_train, y_train, X_test, y_test, tags):
    with mlflow.start_run(run_name=model_name) as run:
        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            try:
                y_proba = model.decision_function(X_test)
            except Exception:
                y_proba = None

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")

        # ROC-AUC (multi-class, if possible)
        roc_auc = None
        try:
            lb = LabelBinarizer()
            lb.fit(y_test)
            y_test_bin = lb.transform(y_test)
            if y_proba is not None:
                if y_proba.ndim == 1 or y_proba.shape[1] == 1:
                    roc_auc = roc_auc_score(y_test_bin, y_proba)
                else:
                    roc_auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
        except Exception:
            roc_auc = None

        # Log params, metrics, tags
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(model.get_params())
        mlflow.set_tags(tags)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        if roc_auc is not None:
            mlflow.log_metric("roc_auc", roc_auc)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_path = os.path.join(APP_DIR, f"cm_{model_name}.png")
        plot_confusion_matrix(cm, np.unique(y_test), model_name, cm_path)
        mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")

        # Classification report
        report = classification_report(y_test, y_pred)
        report_path = os.path.join(APP_DIR, f"report_{model_name}.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path, artifact_path="classification_report")

        # Model file
        model_file = os.path.join(APP_DIR, f"{model_name}.joblib")
        joblib.dump(model, model_file)
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(model_file, artifact_path="model_file")
        return {
            "run_id": run.info.run_id,
            "model": model,
            "metrics": {
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "precision": precision,
                "recall": recall,
                "roc_auc": roc_auc
            },
            "model_file": model_file
        }

def main():
    # Set experiment before any runs
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    print(f"[MLflow] Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"[MLflow] Experiment ID: {experiment.experiment_id if experiment else 'Not found'}")

    # Train all models (each has its own mlflow.start_run() context in log_model_run)
    tags = {"version": VERSION}
    results = {}
    for name, model in get_models().items():
        print(f"Training {name}...")
        result = log_model_run(name, model, X_train, y_train, X_test, y_test, tags)
        results[name] = result

    # Select best model by F1-macro
    best_name = max(results, key=lambda n: results[n]["metrics"]["f1_macro"])
    best = results[best_name]

    # Save best model locally
    joblib.dump(best["model"], MODEL_PATH)

    # Save metadata
    meta = {
        "best_model": best_name,
        "metrics": {
            "accuracy": round(best["metrics"]["accuracy"], 3),
            "f1_macro": round(best["metrics"]["f1_macro"], 3)
        },
        "mlflow_run_id": best["run_id"],
        "version": VERSION
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    # Register best model in MLflow Model Registry
    print(f"Registering best model: {best_name}")
    mlflow_client = mlflow.tracking.MlflowClient()
    try:
        mlflow_client.create_registered_model(REGISTERED_MODEL_NAME)
    except Exception:
        pass  # Already exists
    mlflow_client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=mlflow.get_artifact_uri("model"),
        run_id=best["run_id"]
    )

    print(f"Best model: {best_name} (F1-macro: {best['metrics']['f1_macro']:.3f})")
    print(f"Meta written to {META_PATH}")
    print(f"Model written to {MODEL_PATH}")

if __name__ == "__main__":
    main()
