from fightClassifier.utils.read_yaml import read_yaml
from fightClassifier.entity.param_entity import (DataParam,
                                                 ViViTArchitectureParam,
                                                 TrainingParam,
                                                 OptimizerParam,
                                                 TubeletEmbeddingParam)


from fightClassifier import logger

class ParamManager:
    def __init__(self,PARAMS_PATH='params.yaml'):
        self.params = read_yaml(PARAMS_PATH)

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

