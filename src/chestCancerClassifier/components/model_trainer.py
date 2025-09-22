import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from chestCancerClassifier.entity.config_entity import TrainingConfig

from tensorflow.keras.callbacks import ReduceLROnPlateau
from pathlib import Path
# Set TensorFlow to run eagerly. This can be useful for debugging but may slow down training.
tf.config.run_functions_eagerly(True)

# Define the Training class for the model training pipeline
class Training:
    # Constructor to initialize the class with a configuration object
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None

    # Method to load the pre-trained base model
    def get_base_model(self):
        # The model's architecture and weights are loaded from the specified path
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    # A dedicated method to re-compile the loaded model. This is crucial as
    # the optimizer's state is not saved with the model.
    def compile_model(self):
        self.model.compile(
            # Use the SGD optimizer with the learning rate from the config
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.config.params_learning_rate),
            # Use CategoricalCrossentropy for multi-class classification
            loss=tf.keras.losses.CategoricalCrossentropy(),
            # Track accuracy during training
            metrics=["accuracy"]
        )

    # Method to create and configure data generators for training and validation
    def train_valid_generator(self):
        # Common data augmentation and preprocessing parameters
        datagenerator_kwargs = dict(
            rescale = 1./255, # Scale pixel values to a [0, 1] range
            validation_split= 0.20 # Reserve 20% of data for validation
        )
        # Parameters for the flow_from_directory method
        dataflow_kwargs = dict(
            target_size = self.config.params_image_size[:-1], # Get the image dimensions (height, width)
            batch_size= self.config.params_batch_size, # Set the batch size
            interpolation="bilinear" # Method for resizing images
        )
        
        # Create a data generator for the validation set (no augmentation)
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        # Create the validation data generator from the directory
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory = self.config.training_data,
            subset = "validation",
            shuffle = False, # Do not shuffle validation data
            **dataflow_kwargs
        )

        # Check if data augmentation should be applied to the training set
        if self.config.params_is_augmentation:
            # Create a training data generator with augmentation
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            # If no augmentation, use the same generator as the validation set
            train_datagenerator=valid_datagenerator
        
        # Create the training data generator from the directory
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True, # Shuffle training data
            **dataflow_kwargs
        )
    
    # Static method to save the trained model
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    
    # Method to train the model
    def train(self):
        # Calculate steps per epoch for training and validation
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        
        # Access the nested parameters for the callback from the config
        callback_params = self.config.params_callbacks.ReduceLROnPlateau

        # Define the ReduceLROnPlateau callback with parameters from the config
        lr_scheduler = ReduceLROnPlateau(
            monitor=callback_params.monitor,
            factor=callback_params.factor,
            patience=callback_params.patience,
            min_lr=float(callback_params.min_lr)
        )

        # Train the model using the data generators and defined callbacks
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps = self.validation_steps,
            validation_data = self.valid_generator,
            callbacks=[lr_scheduler] # Pass the callback in a list
        )

        # Save the final trained model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )