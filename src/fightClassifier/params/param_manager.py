from fightClassifier.utils.read_yaml import read_yaml
from fightClassifier.entity.param_entity import (DataParam,
                                                 ViViTArchitectureParam,
                                                 TrainingParam,
                                                 OptimizerParam,
                                                 TubeletEmbeddingParam,
                                                 MeraParam)

import ast


from fightClassifier import logger

class ParamManager:
    def __init__(self,PARAMS_PATH='params.yaml'):
        self.params_value = read_yaml(PARAMS_PATH)
        self.params = self.evaluate_nested_dict(self.params_value)
    
    def safe_eval(self,value):
        try:
            # Use ast.literal_eval for safe evaluation of literals
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Return the original value if it can't be evaluated
            return value

    def evaluate_nested_dict(self,d):
        for key, value in d.items():
            if isinstance(value, str):
                d[key] = self.safe_eval(value)
            elif isinstance(value, dict):
                d[key] = self.evaluate_nested_dict(value)
        return d
    
    def param_data(self)->DataParam:
        try:
            param = self.params['data']
            return DataParam(
                dataset_name=param['DATASET_NAME'],
                batch_size=param['BATCH_SIZE'],
                input_shape=param['INPUT_SHAPE'],
                num_classes=param['NUM_CLASSES']
            )

        except Exception as e:
            logger.error(e)
    
    def param_optimizer(self)->OptimizerParam:
        try:
            param = self.params['optimizer']
            return OptimizerParam(
                learning_rate=param['LEARNING_RATE'],
                weight_decay=param['WEIGHT_DECAY']
            )
        except Exception as e:
            logger.error(e)

    def param_training(self)->TrainingParam:
        try:
            param = self.params['training']
            return TrainingParam(
                epochs=param['EPOCHS']
            )
        except Exception as e:
            logger.error(e)
    
    def param_tubelet_embedding(self)->TubeletEmbeddingParam:
        try:
            param = self.params['tubelet_embedding']
            return TubeletEmbeddingParam(
                patch_size=param['PATCH_SIZE']
            )
        except Exception as e:
            logger.error(e)

    def param_vivit_architecture(self)->ViViTArchitectureParam:
        try:
            param = self.params['vivit_architecture']
            return ViViTArchitectureParam(
                layer_norm_eps=param['LAYER_NORM_EPS'],
                projection_dim=param['PROJECTION_DIM'],
                num_heads=param['NUM_HEADS'],
                num_layers=param['NUM_LAYERS']
            )
        except Exception as e:
            logger.error(e)
        
    def param_mega(self)->MeraParam:
        try:
            return MeraParam(
                data_param=self.param_data(),
                optimizer_param=self.param_optimizer(),
                training_param=self.param_training(),
                tube_embedding_param=self.param_tubelet_embedding(),
                vivit_arch_param=self.param_vivit_architecture()
            )
        except Exception as e:
            logger.error(e)
    

