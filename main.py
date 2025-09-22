import os
from chestCancerClassifier import logger
from chestCancerClassifier.pipeline.stage_01_data_ingestion import DataIngestionTraningPipeline
from chestCancerClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTranningPipline
from chestCancerClassifier.pipeline.stage_03_modeltrainer import ModelTraningPipeline
from chestCancerClassifier.pipeline.stage_04_model_evalution import ModelEvalutationPipeline

os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/proshanta000/End_to_End_ml_project_chest_CT_scan.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="proshanta000"
os.environ["MLFLOW_TRACKING_PASSWORD"]="d856c7bfcbe6c5c979320b3160b26a5a3e1f4355"


"""STAGE_NAME = "Data Ingestion Stage"


if __name__=='__main__':
    try:
        logger.info(f"**********************")
        logger.info(f">>>>>>>>>>  stage {STAGE_NAME} started <<<<<<<<<<")
        obj = DataIngestionTraningPipeline()
        obj.main()
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<\n\nX============X")
    except Exception as e:
        logger.exception(e)
        raise e
    


STAGE_NAME = "Prepare Base Model"

try:
    logger.info(f"**********************")
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} Started <<<<<<<<<<<")
    obj= PrepareBaseModelTranningPipline()
    obj.main()
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<<<\n\nX============X")

except Exception as e:
    logger.exception(e)
    raise e"""


"""STAGE_NAME = "Training"

try:
    logger.info(f"**********************")
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} Started <<<<<<<<<<<")
    obj= ModelTraningPipeline()
    obj.main()
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<<<\n\nX============X")

except Exception as e:
    logger.exception(e)
    raise e"""


STAGE_NAME = "Model Evaluation"

try:
    logger.info(f"**********************")
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} Started <<<<<<<<<<<")
    obj= ModelEvalutationPipeline()
    obj.main()
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<<<\n\nX============X")

except Exception as e:
    logger.exception(e)
    raise e

