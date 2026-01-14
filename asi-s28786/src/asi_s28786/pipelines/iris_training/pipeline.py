from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    prepare_data_splits,
    create_random_forest_model,
    create_logistic_regression_model,
    create_svm_model,
    create_knn_model,
    train_single_model,
    select_best_model,
    save_best_model,
    save_model_metadata,
    extract_run_id_from_results,
    register_model_in_mlflow
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the iris training pipeline.

    Pipeline flow:
    1. Prepare data splits
    2. Create model instances
    3. Train multiple models in parallel
    4. Select best model
    5. Save model and metadata
    6. Register in MLflow
    """
    return pipeline([
        # Data preparation
        node(
            func=prepare_data_splits,
            inputs={
                "iris_raw": "iris_raw",
                "test_size": "params:test_size",
                "random_state": "params:random_state"
            },
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="prepare_data_splits",
        ),

        # Create model instances
        node(
            func=create_random_forest_model,
            inputs={"random_state": "params:random_state"},
            outputs="models.random_forest",
            name="create_random_forest_model",
        ),
        node(
            func=create_logistic_regression_model,
            inputs={"random_state": "params:random_state"},
            outputs="models.logistic_regression",
            name="create_logistic_regression_model",
        ),
        node(
            func=create_svm_model,
            inputs={"random_state": "params:random_state"},
            outputs="models.svm",
            name="create_svm_model",
        ),
        node(
            func=create_knn_model,
            inputs=None,
            outputs="models.knn",
            name="create_knn_model",
        ),

        # Train models in parallel
        node(
            func=train_single_model,
            inputs={
                "model_name": "params:models.random_forest.name",
                "model": "models.random_forest",
                "X_train": "X_train",
                "y_train": "y_train",
                "X_test": "X_test",
                "y_test": "y_test",
                "output_dir": "params:output_dir",
                "experiment_name": "params:mlflow.experiment_name",
                "tags": "params:mlflow.tags"
            },
            outputs="random_forest_results",
            name="train_random_forest",
        ),
        node(
            func=train_single_model,
            inputs={
                "model_name": "params:models.logistic_regression.name",
                "model": "models.logistic_regression",
                "X_train": "X_train",
                "y_train": "y_train",
                "X_test": "X_test",
                "y_test": "y_test",
                "output_dir": "params:output_dir",
                "experiment_name": "params:mlflow.experiment_name",
                "tags": "params:mlflow.tags"
            },
            outputs="logistic_regression_results",
            name="train_logistic_regression",
        ),
        node(
            func=train_single_model,
            inputs={
                "model_name": "params:models.svm.name",
                "model": "models.svm",
                "X_train": "X_train",
                "y_train": "y_train",
                "X_test": "X_test",
                "y_test": "y_test",
                "output_dir": "params:output_dir",
                "experiment_name": "params:mlflow.experiment_name",
                "tags": "params:mlflow.tags"
            },
            outputs="svm_results",
            name="train_svm",
        ),
        node(
            func=train_single_model,
            inputs={
                "model_name": "params:models.knn.name",
                "model": "models.knn",
                "X_train": "X_train",
                "y_train": "y_train",
                "X_test": "X_test",
                "y_test": "y_test",
                "output_dir": "params:output_dir",
                "experiment_name": "params:mlflow.experiment_name",
                "tags": "params:mlflow.tags"
            },
            outputs="knn_results",
            name="train_knn",
        ),

        # Select best model
        node(
            func=select_best_model,
            inputs={
                "random_forest_results": "random_forest_results",
                "logistic_regression_results": "logistic_regression_results",
                "svm_results": "svm_results",
                "knn_results": "knn_results",
                "metric": "params:selection_metric"
            },
            outputs=["best_model_name", "best_model_results"],
            name="select_best_model",
        ),

        # Save best model
        node(
            func=save_best_model,
            inputs={
                "best_model_name": "best_model_name",
                "best_model_results": "best_model_results",
                "output_path": "params:model_output_path",
                "version": "params:version"
            },
            outputs="model_save_path",
            name="save_best_model_locally",
        ),

        # Save metadata
        node(
            func=save_model_metadata,
            inputs={
                "best_model_name": "best_model_name",
                "best_model_results": "best_model_results",
                "output_path": "params:metadata_output_path",
                "version": "params:version"
            },
            outputs="metadata_save_path",
            name="save_model_metadata",
        ),

        # Extract run_id for MLflow registration
        node(
            func=extract_run_id_from_results,
            inputs={"best_model_results": "best_model_results"},
            outputs="mlflow_run_id",
            name="extract_run_id",
        ),

        # Register in MLflow
        node(
            func=register_model_in_mlflow,
            inputs={
                "model_name": "best_model_name",
                "run_id": "mlflow_run_id",
                "registered_model_name": "params:mlflow.registered_model_name"
            },
            outputs=None,
            name="register_model_in_mlflow",
        ),
    ])