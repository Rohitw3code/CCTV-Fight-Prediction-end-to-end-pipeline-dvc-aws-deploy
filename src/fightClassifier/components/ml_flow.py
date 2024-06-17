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
        eval_ = load_json(self.config.evaluation_param_path)
        model_param = load_json(self.config.model_param_path)
        model = ModelTraining().load_model()


        mlflow.end_run()

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # End any existing run if it exists
        if mlflow.active_run() is not None:
            mlflow.end_run()

        # Start a new MLflow run
        with mlflow.start_run(run_name=self.config.project_name):
            mlflow.log_params(model_param)
            mlflow.log_metrics(eval_)
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(model, "model", registered_model_name=self.config.model_name)
            else:
                mlflow.keras.log_model(model, "model")

