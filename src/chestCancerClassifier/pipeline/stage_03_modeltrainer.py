from chestCancerClassifier.config.configuration import ConfigurationManager
from chestCancerClassifier.components.model_trainer import Training
from chestCancerClassifier import logger



STAGE_NAME = "Training"

class ModelTraningPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_traning_config()
        training = Training(config=training_config)
        
        # 1. Load the model's architecture and weights.
        training.get_base_model() 
        
        # 2. Re-compile the model. This is the crucial step.
        training.compile_model() 
        
        # 3. Set up the data generators.
        training.train_valid_generator()
        
        # 4. Start the training process.
        training.train()


if __name__=='__main__':
    try:
        logger.info(f"**********************")
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} Started <<<<<<<<<<<")
        obj= ModelTraningPipeline()
        obj.main()
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<<<\n\nX============X")

    except Exception as e:
        logger.exception(e)
        raise e