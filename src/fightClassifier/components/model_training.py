from fightClassifier.entity.config_entity import ModelTrainConfig
from fightClassifier.components.Encode import TubeletEmbedding,PositionalEncoder
from fightClassifier.config.configuration import ConfigurationManager
from fightClassifier.entity.param_entity import MeraParam
from fightClassifier.utils import save_json
from fightClassifier.config.configuration import ConfigurationManager
from keras import layers
import tensorflow as tf
import keras
from keras import ops
import os


class ModelTraining:
    def __init__(self,trainLoader=None,
                 testLoader=None,
                 validLoader=None,
                 params:MeraParam=dict()):
        self.trainloader = trainLoader
        self.testloader = testLoader
        self.validloader = validLoader
        self.params = params
        self.config = ConfigurationManager()
        self.config = self.config.config_model_train()
        self.model = None

    def _create_vivit_classifier(
        self,
        tubelet_embedder,
        positional_encoder):
        # Get the input layer
        inputs = layers.Input(shape=self.params.data_param.input_shape)
        # Create patches.
        patches = tubelet_embedder(inputs)
        # Encode patches.
        encoded_patches = positional_encoder(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(self.params.vivit_arch_param.num_layers):
            # Layer normalization and MHSA
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = layers.MultiHeadAttention(
                num_heads=self.params.vivit_arch_param.num_heads,
                key_dim=self.params.vivit_arch_param.projection_dim // self.params.vivit_arch_param.num_heads, 
                dropout=0.1
            )(x1, x1)

            # Skip connection
            x2 = layers.Add()([attention_output, encoded_patches])

            # Layer Normalization and MLP
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = keras.Sequential(
                [
                    layers.Dense(units=self.params.vivit_arch_param.projection_dim * 4, activation=ops.gelu),
                    layers.Dense(units=self.params.vivit_arch_param.projection_dim, activation=ops.gelu),
                ]
            )(x3)

            # Skip connection
            encoded_patches = layers.Add()([x3, x2])

        # Layer normalization and Global average pooling.
        representation = layers.LayerNormalization(epsilon=self.params.vivit_arch_param.layer_norm_eps)(encoded_patches)
        representation = layers.GlobalAvgPool1D()(representation)

        # Classify outputs.
        outputs = layers.Dense(units=self.params.data_param.num_classes,
                            activation="softmax")(representation)

        # Create the Keras model.
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def train(self):
        # Initialize model
        self.model = self._create_vivit_classifier(
            tubelet_embedder=TubeletEmbedding(
                embed_dim=self.params.vivit_arch_param.projection_dim,
                patch_size=self.params.tube_embedding_param.patch_size
            ),
            positional_encoder=PositionalEncoder(embed_dim=self.params.vivit_arch_param.projection_dim),
        )

        # Compile the model with the optimizer, loss function
        # and the metrics.
        optimizer = keras.optimizers.Adam(learning_rate=self.params.optimizer_param.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
        )

        # print('train size : ',self.trainloader.shape)

        # Train the model.
        _ = self.model.fit(self.trainloader, epochs=self.params.training_param.epochs, validation_data=self.validloader)

        return self.model
    
    def mlflow_tracker(self):
        mlflow_config = ConfigurationManager().config_mlflow()
        save_json(mlflow_config.model_param_path,{
                'batch_size':self.params.data_param.batch_size,
                'patch_size':self.params.tube_embedding_param.patch_size,
                'epochs':self.params.training_param.epochs,
                'learning_rate':self.params.optimizer_param.learning_rate,
                'weight_decay':self.params.optimizer_param.weight_decay,
            })

    
    def load_model(self):
        self.model = tf.keras.models.load_model(os.path.join(self.config.save_model_dir,
                                     self.config.save_model_name))
        return self.model

    def save_model(self):
        os.makedirs(self.config.save_model_dir)
        self.model.save(os.path.join(self.config.save_model_dir,
                                     self.config.save_model_name))