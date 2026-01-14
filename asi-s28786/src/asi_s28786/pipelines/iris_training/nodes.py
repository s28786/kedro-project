import os
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelBinarizer


def get_model_zoo() -> Dict[str, Any]:
    """
    Returns a dictionary of classifier models to train.

    Returns:
        Dictionary mapping model names to sklearn classifier instances.
    """
    return {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=200, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier()
    }


def plot_confusion_matrix(
        cm: np.ndarray,
        classes: np.ndarray,
        model_name: str,
        out_path: str
) -> None:
    """
    Plot and save confusion matrix as a PNG file.

    Args:
        cm: Confusion matrix array
        classes: Class labels
        model_name: Name of the model for the title
        out_path: Path where to save the plot
    """
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


def calculate_roc_auc(y_test, y_proba) -> float:
    """
    Calculate ROC-AUC score for multi-class classification.

    Args:
        y_test: True labels
        y_proba: Predicted probabilities or decision function output

    Returns:
        ROC-AUC score or None if calculation fails
    """
    try:
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test_bin = lb.transform(y_test)
        if y_proba is not None:
            if y_proba.ndim == 1 or y_proba.shape[1] == 1:
                return roc_auc_score(y_test_bin, y_proba)
            else:
                return roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
    except Exception:
        return None


def train_single_model(
        model_name: str,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        output_dir: str,
        experiment_name: str,
        tags: Dict[str, str]
) -> Dict[str, Any]:
    """
    Train a single model, log metrics and artifacts to MLflow.

    Note: MLflow run is managed by kedro-mlflow, so we don't create our own run context.

    Args:
        model_name: Name of the model
        model: Sklearn classifier instance
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        output_dir: Directory to save artifacts
        experiment_name: MLflow experiment name (not used, managed by kedro-mlflow)
        tags: Tags to attach to the MLflow run

    Returns:
        Dictionary containing run_id, trained model, metrics, and model file path
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Get predicted probabilities
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        try:
            y_proba = model.decision_function(X_test)
        except Exception:
            y_proba = None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    roc_auc = calculate_roc_auc(y_test, y_proba)

    # Log parameters, metrics, and tags to the active MLflow run
    mlflow.log_param(f"{model_name}_name", model_name)
    for param_name, param_value in model.get_params().items():
        mlflow.log_param(f"{model_name}_{param_name}", param_value)

    mlflow.set_tags({f"{model_name}_{k}": v for k, v in tags.items()})
    mlflow.log_metric(f"{model_name}_accuracy", accuracy)
    mlflow.log_metric(f"{model_name}_f1_macro", f1_macro)
    mlflow.log_metric(f"{model_name}_precision", precision)
    mlflow.log_metric(f"{model_name}_recall", recall)
    if roc_auc is not None:
        mlflow.log_metric(f"{model_name}_roc_auc", roc_auc)

    # Save and log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(output_dir, f"cm_{model_name}.png")
    plot_confusion_matrix(cm, np.unique(y_test), model_name, cm_path)
    mlflow.log_artifact(cm_path, artifact_path="confusion_matrices")

    # Save and log classification report
    report = classification_report(y_test, y_pred)
    report_path = os.path.join(output_dir, f"report_{model_name}.txt")
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path, artifact_path="classification_reports")

    # Save and log model - use artifact_path for directory, not model name
    model_file = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(model, model_file)
    # Log model with clean name (no slashes)
    mlflow.sklearn.log_model(model, artifact_path=f"model_{model_name}")
    mlflow.log_artifact(model_file, artifact_path="saved_models")

    # Get run_id from active run
    run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None

    return {
        "run_id": run_id,
        "model": model,
        "metrics": {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc
        },
        "model_file": model_file,
        "model_name": model_name
    }


def select_best_model(
        random_forest_results: Dict[str, Any],
        logistic_regression_results: Dict[str, Any],
        svm_results: Dict[str, Any],
        knn_results: Dict[str, Any],
        metric: str = "f1_macro"
) -> Tuple[str, Dict[str, Any]]:
    """
    Select the best model based on a specified metric.

    Args:
        random_forest_results: Results for RandomForest model
        logistic_regression_results: Results for LogisticRegression model
        svm_results: Results for SVM model
        knn_results: Results for KNN model
        metric: Metric to use for selection (default: f1_macro)

    Returns:
        Tuple of (best_model_name, best_model_results)
    """
    model_results = {
        "RandomForest": random_forest_results,
        "LogisticRegression": logistic_regression_results,
        "SVM": svm_results,
        "KNN": knn_results
    }

    best_name = max(model_results, key=lambda n: model_results[n]["metrics"][metric])

    # Log best model info to MLflow
    mlflow.log_metric("best_model_f1_macro", model_results[best_name]["metrics"][metric])
    mlflow.log_param("best_model_name", best_name)

    return best_name, model_results[best_name]


def save_best_model(
        best_model_name: str,
        best_model_results: Dict[str, Any],
        output_path: str,
        version: str = "v1.0.0"
) -> str:
    """
    Save the best model to a joblib file.

    Args:
        best_model_name: Name of the best model
        best_model_results: Results dictionary for the best model
        output_path: Path where to save the model
        version: Model version string

    Returns:
        Path to the saved model file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(best_model_results["model"], output_path)
    print(f"Best model ({best_model_name}) saved to {output_path}")

    # Log final model to MLflow
    mlflow.sklearn.log_model(best_model_results["model"], artifact_path="best_model")

    return output_path


def save_model_metadata(
        best_model_name: str,
        best_model_results: Dict[str, Any],
        output_path: str,
        version: str = "v1.0.0"
) -> str:
    """
    Save model metadata to a JSON file.

    Args:
        best_model_name: Name of the best model
        best_model_results: Results dictionary for the best model
        output_path: Path where to save the metadata
        version: Model version string

    Returns:
        Path to the saved metadata file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    meta = {
        "best_model": best_model_name,
        "metrics": {
            "accuracy": round(best_model_results["metrics"]["accuracy"], 3),
            "f1_macro": round(best_model_results["metrics"]["f1_macro"], 3),
            "precision": round(best_model_results["metrics"]["precision"], 3),
            "recall": round(best_model_results["metrics"]["recall"], 3)
        },
        "mlflow_run_id": best_model_results["run_id"],
        "version": version
    }

    if best_model_results["metrics"]["roc_auc"] is not None:
        meta["metrics"]["roc_auc"] = round(best_model_results["metrics"]["roc_auc"], 3)

    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Model metadata saved to {output_path}")

    # Log metadata as artifact
    mlflow.log_artifact(output_path, artifact_path="metadata")

    return output_path


def register_model_in_mlflow(
        model_name: str,
        run_id: str,
        registered_model_name: str = "IrisModel"
) -> None:
    """
    Register the best model in MLflow Model Registry.

    Args:
        model_name: Name of the model being registered
        run_id: MLflow run ID
        registered_model_name: Name to use in the model registry
    """
    if run_id is None:
        print("No run_id available, skipping model registration")
        return

    mlflow_client = mlflow.tracking.MlflowClient()

    # Create registered model if it doesn't exist
    try:
        mlflow_client.create_registered_model(registered_model_name)
        print(f"Created new registered model: {registered_model_name}")
    except Exception:
        print(f"Registered model {registered_model_name} already exists")

    # Create model version
    model_uri = f"runs:/{run_id}/best_model"
    mlflow_client.create_model_version(
        name=registered_model_name,
        source=model_uri,
        run_id=run_id
    )
    print(f"Registered {model_name} in MLflow Model Registry as {registered_model_name}")


def prepare_data_splits(
        iris_raw: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the Iris dataset into training and test sets.

    Args:
        iris_raw: Raw Iris dataframe with columns Id, Species, and features
        test_size: Proportion of test data (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = iris_raw.drop(["Id", "Species"], axis=1)
    y = iris_raw["Species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def create_random_forest_model(
        n_estimators: int = 100,
        max_depth: int = None,
        random_state: int = 42
) -> RandomForestClassifier:
    """
    Create RandomForest classifier instance with hyperparameters.

    Args:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the tree
        random_state: Random seed for reproducibility

    Returns:
        RandomForestClassifier instance
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )


def create_logistic_regression_model(
        max_iter: int = 200,
        random_state: int = 42
) -> LogisticRegression:
    """
    Create LogisticRegression classifier instance with hyperparameters.

    Args:
        max_iter: Maximum number of iterations
        random_state: Random seed for reproducibility

    Returns:
        LogisticRegression instance
    """
    return LogisticRegression(max_iter=max_iter, random_state=random_state)


def create_svm_model(
        C: float = 1.0,
        probability: bool = True,
        random_state: int = 42
) -> SVC:
    """
    Create SVM classifier instance with hyperparameters.

    Args:
        C: Regularization parameter
        probability: Whether to enable probability estimates
        random_state: Random seed for reproducibility

    Returns:
        SVC instance
    """
    return SVC(C=C, probability=probability, random_state=random_state)


def create_knn_model(n_neighbors: int = 5) -> KNeighborsClassifier:
    """
    Create KNN classifier instance with hyperparameters.

    Args:
        n_neighbors: Number of neighbors to use

    Returns:
        KNeighborsClassifier instance
    """
    return KNeighborsClassifier(n_neighbors=n_neighbors)


def extract_run_id_from_results(best_model_results: Dict[str, Any]) -> str:
    """
    Extract run_id from best model results for MLflow registration.

    Args:
        best_model_results: Dictionary containing model results with run_id

    Returns:
        MLflow run ID string
    """
    return best_model_results["run_id"]