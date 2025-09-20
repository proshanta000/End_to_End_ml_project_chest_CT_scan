from chestCancerClassifier import logger
from chestCancerClassifier.pipeline.stage_01_data_ingestion import DataIngestionTraningPipeline


STAGE_NAME = "Data Ingestion Stage"


if __name__=='__main__':
    try:
        logger.info(f">>>>>>  stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTraningPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx========X")
    except Exception as e:
        logger.exception(e)
        raise e