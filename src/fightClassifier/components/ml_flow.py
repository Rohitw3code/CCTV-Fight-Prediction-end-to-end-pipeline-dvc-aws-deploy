from fightClassifier.config.configuration import MLFlowConfig
from fightClassifier.utils.read_yaml import load_json
from fightClassifier.components.model_training import ModelTraining
import dagshub
import os
from urllib.parse import urlparse
import mlflow


class MLFlowSetUp:
    def __init__(self,config:MLFlowConfig):
        self.config = config

    def init(self):
        mlflow.set_registry_uri(self.config.mlflow_tracking_uri)
        dagshub.init(repo_owner=self.config.repo_owner,
                    repo_name=self.config.repo_name,
                    mlflow=self.config.mlflow)

    def setup_credentials(self):
        os.environ['MLFLOW_TRACKING_URI'] = self.config.mlflow_tracking_uri
        os.environ['MLFLOW_TRACKING_USERNAME'] = self.config.mlflow_tracking_username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = self.config.mlflow_tracking_password
    
    def mlflow_tracker(self):
        eval_ = load_json('artifacts/mlflow_data/evaluation.json')
        model_param = load_json('artifacts/mlflow_data/model.json')
        model = ModelTraining().load_model()


        mlflow.end_run()

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        while mlflow.start_run():
            mlflow.log_params(model_param)
            mlflow.log_metrics(eval_)
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(model, "model", registered_model_name="video-vision")
            else:
                mlflow.keras.log_model(model, "model")

