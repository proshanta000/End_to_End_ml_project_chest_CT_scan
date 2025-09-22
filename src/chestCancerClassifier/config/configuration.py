import os
from pathlib import Path
from chestCancerClassifier.constants import *
from chestCancerClassifier.utils.common import (read_yaml,
                                               create_directories,
                                               save_json)
from chestCancerClassifier.entity.config_entity import (DataIngestionConfig, 
                                                       PrepareBaseModelConfig,
                                                       TrainingConfig,
                                                       EvaluationConfig)

# This class manages the configuration settings for the entire project pipeline.
# It reads settings from 'config.yaml' and 'params.yaml' and provides them
# to each pipeline stage in a structured format.
class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        """
        Initializes the ConfigurationManager by loading configuration and
        parameters from YAML files.
        """
        # Load the main configuration file
        self.config = read_yaml(config_filepath)
        # Load the model hyperparameters
        self.params = read_yaml(params_filepath)

        # Create the root artifacts directory if it doesn't exist
        create_directories([self.config.artifacts_root])

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Provides the configuration for the data ingestion stage.
        """
        # Get data ingestion settings from the main config
        config = self.config.data_ingestion

        # Create the data ingestion root directory
        create_directories([config.root_dir])

        # Create a DataIngestionConfig object with the relevant settings
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_prepear_base_model_config(self) -> PrepareBaseModelConfig:
        """
        Provides the configuration for the base model preparation stage.
        """
        # Get settings from the main config
        config = self.config.prepare_base_model

        # Create the directory for the base model
        create_directories([config.root_dir])

        # Create a PrepareBaseModelConfig object with all necessary parameters
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir= Path(config.root_dir),
            base_model_path= Path(config.base_model_path),
            updated_base_model_path= Path(config.updated_base_model_path),
            params_image_size= self.params.IMAGE_SIZE,  # Input image dimensions
            params_learning_rate= self.params.LEARNING_RATE, # Learning rate for the optimizer
            params_include_top= self.params.INCLUDE_TOP, # Whether to include the final dense layer
            params_weights= self.params.WEIGHTS, # Pre-trained weights for the model
            params_classes= self.params.CLASSES # The number of output classes, crucial for model architecture
        )

        return prepare_base_model_config
    

    def get_traning_config(self) -> TrainingConfig:
        """
        Provides the configuration for the model training stage.
        """
        # Get settings from main config and params files
        traning = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        
        # Define the path to the training data directory
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "chest_CT_scan_data", "train")
        
        # Create the training artifacts directory
        create_directories([Path(traning.root_dir)])

        # Access the entire nested CALLBACKS section from params
        callbacks_params = params.CALLBACKS
        
        # Create a TrainingConfig object with all necessary parameters
        traning_config = TrainingConfig(
            root_dir=Path(traning.root_dir),
            trained_model_path=Path(traning.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS, # Number of training epochs
            params_batch_size=params.BATCH_SIZE, # Batch size for training
            params_is_augmentation=params.AUGMENTATION,  # Enable/disable data augmentation
            params_image_size=params.IMAGE_SIZE,  # Set the input image size
            params_learning_rate=params.LEARNING_RATE,  # Define the learning rate
            
            # Pass the nested parameters to the dataclass
            params_callbacks=callbacks_params  # Define the callbacks for training
        )

        return traning_config
    

    def get_evalution_config(self) -> EvaluationConfig:
        """
        Provides the configuration for the model evaluation stage.
        """
        # Define the path to the validation data directory
        validation_data = os.path.join(self.config.data_ingestion.unzip_dir, "chest_CT_scan_data", "valid")
        
        # Create an EvaluationConfig object with the relevant settings
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5", # Path to the trained model
            validation_data=Path(validation_data), # Path to the validation data
            mlflow_uri="https://dagshub.com/proshanta000/End_to_End_ml_project_chest_CT_scan.mlflow", # MLflow tracking URI
            all_params= self.params, # All model parameters for logging to MLflow
            params_image_size= self.params.IMAGE_SIZE, # Image size for evaluation
            params_batch_size= self.params.BATCH_SIZE # Batch size for evaluation
        )
        return eval_config
