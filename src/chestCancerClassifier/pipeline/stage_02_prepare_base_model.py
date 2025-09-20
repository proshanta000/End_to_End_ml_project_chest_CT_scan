from chestCancerClassifier.config.configuration import ConfigurationManager
from chestCancerClassifier.components.prepare_base_mode import PrepareBaseModel
from chestCancerClassifier import logger



STAGE_NAME = "Prepare base model"

class PrepareBaseModelTranningPipline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepear_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


if __name__=='__main__':
    try:
        logger.info(f"**********************")
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} Started <<<<<<<<<<<")
        obj= PrepareBaseModelTranningPipline()
        obj.main()
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<<<\n\nX============X")

    except Exception as e:
        logger.exception(e)
        raise e