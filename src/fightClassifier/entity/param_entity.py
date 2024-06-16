from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class DataParam:
    dataset_name: str
    batch_size: int
    input_shape: Tuple[int, int, int, int]
    num_classes: int

@dataclass(frozen=True)
class OptimizerParam:
    learning_rate: float
    weight_decay: float

@dataclass(frozen=True)
class TrainingParam:
    epochs: int

@dataclass(frozen=True)
class TubeletEmbeddingParam:
    patch_size: Tuple[int, int, int]

@dataclass(frozen=True)
class ViViTArchitectureParam:
    layer_norm_eps: float
    projection_dim: int
    num_heads: int
    num_layers: int

@dataclass(frozen=True)
class MeraParam:
    data_param:DataParam
    optimizer_param:OptimizerParam
    training_param:TrainingParam
    tube_embedding_param:TubeletEmbeddingParam
    vivit_arch_param:ViViTArchitectureParam
