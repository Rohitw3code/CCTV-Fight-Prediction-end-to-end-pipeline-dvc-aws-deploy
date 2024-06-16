from fightClassifier.config.configuration import MLFlowConfig
import dagshub
import os
import mlflow


class MLFlowSetUp:
    def __init__(self,config:MLFlowConfig):
        self.config = config

    def init(self):
        mlflow.set_registry_uri(self.config.mlflow_tracking_uri)
        dagshub.init(repo_owner=self.config.repo_owner,
                      repo_name=self.config.repo_name
                     , mlflow=self.config.mlflow)

    def setup_credentials(self):
        os.environ['MLFLOW_TRACKING_URI'] = self.config.mlflow_tracking_uri
        os.environ['MLFLOW_TRACKING_USERNAME'] = self.config.mlflow_tracking_username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = self.config.mlflow_tracking_password
