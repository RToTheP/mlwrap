"""Module for Keras algorithms"""

import math
import logging

from mlwrap.enums import ProblemType

early_stopping_iterations: int = 50
batch_size: int = 32
epochs: int = 1000

def get_keras(problem_type: ProblemType, adapt_class_weights: bool = False):
    import tensorflow as tf
    from tensorflow import keras
    from scikeras.wrappers import KerasClassifier, KerasRegressor

    # callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="loss", patience=early_stopping_iterations
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=early_stopping_iterations,
        ),
    ]

    # class weights
    class_weights = None
    if adapt_class_weights and problem_type == ProblemType.Classification:
        class_weights = "balanced"

    if problem_type == ProblemType.Regression:
        return KerasRegressor(
            get_model,
            problem_type=problem_type,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=2,
        )

    if problem_type == ProblemType.Classification:
        return KerasClassifier(
            get_model,
            problem_type=problem_type,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=2,
        )

    raise NotImplementedError


def get_model(problem_type: ProblemType, meta):
    import tensorflow as tf
    from tensorflow import keras
    
    # create a new model from scratch
    n_features_in_ = meta["n_features_in_"]
    input_layer_nodes = n_features_in_

    if problem_type == ProblemType.Regression:
        outputLayerNodes = meta["n_outputs_"]
        hiddenLayerNodes: int = (
            int(round(math.sqrt(input_layer_nodes))) * outputLayerNodes
        )

        model = keras.Sequential(
            [
                keras.layers.Dense(
                    hiddenLayerNodes,
                    activation=tf.nn.relu,
                    input_shape=(input_layer_nodes,),
                ),
                keras.layers.Dense(outputLayerNodes, activation=tf.nn.sigmoid),
            ]
        )
        optimizer = tf.keras.optimizers.RMSprop(0.001)
        model.compile(
            optimizer=optimizer,
            loss="mean_squared_error",
            metrics=["mean_absolute_error"],
        )

    elif problem_type == ProblemType.Classification:
        outputLayerNodes = meta["n_classes_"]
        hiddenLayerNodes: int = (
            int(round(math.sqrt(input_layer_nodes))) * outputLayerNodes
        )

        model = keras.Sequential(
            [
                keras.layers.Dense(
                    hiddenLayerNodes,
                    activation=tf.nn.relu,
                    input_shape=(input_layer_nodes,),
                ),
                keras.layers.Dense(outputLayerNodes, activation=tf.nn.softmax),
            ]
        )
        optimizer = tf.keras.optimizers.Adam()
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["acc"],
        )
    else:
        raise NotImplementedError

    logging.info(f"{problem_type} model with {outputLayerNodes} output layer nodes")
    model.summary()
    return model
