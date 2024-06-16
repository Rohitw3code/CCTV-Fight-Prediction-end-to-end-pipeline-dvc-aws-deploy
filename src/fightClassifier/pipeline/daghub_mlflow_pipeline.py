from fightClassifier.components.ml_flow import MLFlowSetUp
from fightClassifier.config.configuration import ConfigurationManager


class MLFlowPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        config = config.config_mlflow()
        mlflow = MLFlowSetUp(config=config)
        mlflow.init()
        mlflow.setup_credentials()
        mlflow.mlflow_tracker()



if __name__ == '__main__':
    mlflow_pipeline = MLFlowPipeline()
    mlflow_pipeline.main()