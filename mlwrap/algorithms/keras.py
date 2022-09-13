"""Module for Keras algorithms"""

import math
import logging
from threading import Event

import numpy as np
import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasClassifier, KerasRegressor

from io import BytesIO
from h5py import File
from sklearn.utils import class_weight

from mlwrap.algorithms.base import AlgorithmBase
from mlwrap.config import MLConfig, TrainingResults
from mlwrap.data.config import DataDetails
from mlwrap.data.preparation import get_validation_data, flatten_output_data
from mlwrap.enums import ProblemType, AlgorithmType


def get_model(problem_type: ProblemType, meta):
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


def get_keras(config: MLConfig, X_train, X_test, y_train, y_test):
    # callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="loss", patience=config.early_stopping_iterations
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.early_stopping_iterations,
        ),
    ]

    # class weights
    class_weights = None
    if config.adapt_class_weights and config.problem_type == ProblemType.Classification:
        class_weights = class_weight.compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )

    if config.problem_type == ProblemType.Regression:
        return KerasRegressor(
            get_model,
            problem_type=config.problem_type,
            batch_size=config.model_training_batch_size,
            epochs=config.maximum_training_iterations,
            validation_split=0.2,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=2,
        )

    if config.problem_type == ProblemType.Classification:
        return KerasClassifier(
            get_model,
            problem_type=config.problem_type,
            batch_size=config.model_training_batch_size,
            epochs=config.maximum_training_iterations,
            validation_split=0.2,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=2,
        )

    raise NotImplementedError


class KerasEarlyStoppingCallback(keras.callbacks.Callback):
    def __init__(self, stop_event: Event = None):
        self._stop_event = stop_event

    def on_epoch_end(self, epoch, logs=None):
        if self._stop_event is not None and self._stop_event.is_set():
            logging.info("Stopping due to cancellation event")
            self.stopped_epoch = epoch
            self.model.stop_training = True


class KerasNeuralNetwork(AlgorithmBase):
    @property
    def algorithm(self) -> AlgorithmType:
        return AlgorithmType.KerasNeuralNetwork

    def __init__(self, config: MLConfig, stop_event: Event):
        super().__init__(config, stop_event)

        # Check TensorFlow version
        logging.debug(f"TF version: {tf.version.VERSION}")

        if tf.config.list_physical_devices("GPU"):
            logging.debug("GPU available")
        else:
            logging.debug("GPU not available")

    def _build_model(self, data_details: DataDetails):
        # create a new model from scratch
        inputLayerNodes: int = data_details.train_input.shape[1]
        outputLayerNodes: int = data_details.train_output.shape[1]
        hiddenLayerNodes: int = (
            int(round(math.sqrt(inputLayerNodes))) * outputLayerNodes
        )
        logging.info(
            f"{self._config.problem_type} model with {outputLayerNodes} output layer nodes"
        )
        if self._config.problem_type == ProblemType.Regression:
            model = keras.Sequential(
                [
                    keras.layers.Dense(
                        hiddenLayerNodes,
                        activation=tf.nn.relu,
                        input_shape=(inputLayerNodes,),
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

        elif self._config.problem_type == ProblemType.Classification:
            model = keras.Sequential(
                [
                    keras.layers.Dense(
                        hiddenLayerNodes,
                        activation=tf.nn.relu,
                        input_shape=(inputLayerNodes,),
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

        model.summary()
        self._model = model

    def fit(self, data_details: DataDetails) -> TrainingResults:
        # we build the model before calling flatten_output_data because we need to know the number of output labels
        # this is because we use sparse categorical cross-entropy and want the output layer to be a vector of probabilities
        if self._model is None:
            self._build_model(data_details)

        train_output_flat, test_output_flat = flatten_output_data(
            self._config.problem_type,
            data_details.train_output,
            data_details.test_output,
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="loss", patience=self._config.early_stopping_iterations
            ),
            KerasEarlyStoppingCallback(self._stop_event),
        ]

        validation_data = get_validation_data(data_details.test_input, test_output_flat)

        if validation_data is not None:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=self._config.early_stopping_iterations,
                )
            )

        class_weight = (
            data_details.class_weights if self._config.adapt_class_weights else None
        )

        self._model.fit(
            data_details.train_input,
            train_output_flat,
            batch_size=self._config.model_training_batch_size,
            epochs=self._config.maximum_training_iterations,
            validation_data=validation_data,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=2,
        )
        self.iterations_ = len(self._model.history.epoch)

        return self.get_training_results(data_details=data_details)

    def predict(self, data):
        return self._model.predict(data, verbose=0)

    def get_model_bytes(self) -> bytes:
        binary_image = None
        with File(
            "does not matter", mode="w", driver="core", backing_store=False
        ) as h5file:
            self._model.save(h5file)
            # Very important! Otherwise you get all zeroes below.
            h5file.flush()
            binary_image = h5file.id.get_file_image()

        return binary_image

    def load_model(self, model_bytes: bytes = None):
        path_or_bytes = None
        if isinstance(model_bytes, bytes):
            path_or_bytes = File(BytesIO(model_bytes), "r")

        model = keras.models.load_model(path_or_bytes, compile=False)
        model.summary()
        return model
