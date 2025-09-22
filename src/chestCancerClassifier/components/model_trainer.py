import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from collections import Counter
import time
from chestCancerClassifier.entity.config_entity import TrainingConfig

from tensorflow.keras.callbacks import ReduceLROnPlateau
from pathlib import Path
# Note: Eager execution is useful for debugging but should be disabled for performance in production.
# tf.config.run_functions_eagerly(True)


# This function is a simple utility to get a list of all images in the dataset
def get_image_paths(data_dir):
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image_paths.append(os.path.join(root, file))
    return image_paths

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        # Load the pre-trained base model from the specified path
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
    
    # A dedicated method to compile the model.
    def compile_model(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.config.params_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    def train_valid_generator(self):
        # Define the image data generators for training and validation
        datagenerator_kwargs = dict(
            rescale=1./255
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        
        # We no longer need the validation_split as we are using separate folders
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # Point the generator directly to the 'valid' folder
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=os.path.join(self.config.training_data, "valid"),
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
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
            train_datagenerator = valid_datagenerator

        # Point the generator directly to the 'train' folder
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=os.path.join(self.config.training_data, "train"),
            shuffle=True,
            **dataflow_kwargs
        )
        
        print("Classes and their indices found by the generator:", self.train_generator.class_indices)


    # This is the new function to calculate class weights
    def calculate_class_weights(self):
        # Get the class indices from the training generator
        labels = self.train_generator.labels
        
        # Count the occurrences of each class
        class_counts = Counter(labels)
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)
        
        # Calculate the class weights using the inverse of the class frequency
        class_weights = {}
        for cls_id, count in class_counts.items():
            class_weights[cls_id] = total_samples / (num_classes * count)
        
        print(f"Calculated class weights: {class_weights}")
        return class_weights

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        # The model must be compiled before training
        self.compile_model()
        
        # Calculate the class weights
        class_weights = self.calculate_class_weights()

        # Define the callbacks
        callbacks = [
            # Removed EarlyStopping to ensure the model trains for the full number of epochs
            tf.keras.callbacks.ModelCheckpoint(self.config.trained_model_path, save_best_only=True)
        ]

        # Fit the model with the class weights
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            callbacks=callbacks,
            class_weight=class_weights  # The class weights are applied here
        )
        
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
