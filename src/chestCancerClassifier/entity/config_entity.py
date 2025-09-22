from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration for the data ingestion stage.
    """
    root_dir: Path  # Root directory where data ingestion artifacts are stored
    source_URL: str  # URL to the source data file
    local_data_file: Path  # Local path to save the downloaded data file
    unzip_dir: Path  # Directory where the data will be unzipped

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    """
    Configuration for preparing the base model (VGG16).
    """
    root_dir: Path  # Root directory for this pipeline stage
    base_model_path: Path  # Path to the original VGG16 base model file
    updated_base_model_path: Path  # Path to the updated base model with a custom top layer
    params_image_size: list  # Dimensions of the input images
    params_learning_rate: float  # Learning rate for the optimizer
    params_include_top: bool  # Whether to include the final dense layer of the base model
    params_weights: str  # Pre-trained weights to use for the base model
    params_classes: int  # The number of output classes for the final layer

@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration for the model training stage.
    """
    root_dir: Path  # Root directory for training artifacts
    trained_model_path: Path  # Path to save the final trained model
    updated_base_model_path: Path  # Path to the base model from the preparation stage
    training_data: Path  # Path to the training dataset
    params_epochs: int  # Number of training epochs
    params_batch_size: int  # Number of samples per batch
    params_is_augmentation: bool  # Flag to enable/disable data augmentation
    params_image_size: list  # Image dimensions for training
    params_learning_rate: float  # Learning rate for the optimizer
    params_callbacks: Dict[str, Any]  # Dictionary of callbacks (e.g., EarlyStopping, ReduceLROnPlateau)

@dataclass(frozen=True)
class EvaluationConfig:
    """
    Configuration for the model evaluation stage.
    """
    path_of_model: Path  # Path to the trained model to be evaluated
    validation_data: Path  # Path to the validation dataset
    mlflow_uri: str  # MLflow tracking server URI
    all_params: dict  # A dictionary containing all project parameters for logging
    params_image_size: list  # Image dimensions for evaluation
    params_batch_size: int  # Batch size for evaluation
