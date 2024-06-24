from model.IModel import IModel
from model.KerasModel import KerasModel
from model.ModelBuilderUtils import getLoss, getMetrics, getFedCoreOptimizers, getOptimizer
from utils.Utils import Utils

import attrs
import logging
import tensorflow as tf
import tensorflow_federated as tff
from typing import Any

class FedCoreModel(IModel):
    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger("model/FedCoreModel")
        self.logger.setLevel(config["log_level"])

    @classmethod
    def createFedModel(self_class, fed_data, config):
        keras_model = KerasModel.createKerasModel(fed_data[0], config)
        fed_model = tff.learning.models.from_keras_model(
            keras_model = keras_model,
            input_spec = fed_data[0].element_spec,
            loss = getLoss(config),
            metrics = getMetrics(config))
        return fed_model

    def fit(self, fed_dataset):
        self.logger.info(f'Fitting federated model with {self.config["num_workers"]} workers')

        def cfm():
            keras_model = KerasModel.createKerasModel(fed_dataset.train[0], self.config)
            fed_model = tff.learning.models.from_keras_model(
                keras_model,
                input_spec = fed_dataset.train[0].element_spec,
                loss = getLoss(self.config),
                metrics = getMetrics(self.config))
            return fed_model

        server_optimizer, client_optimizer = getFedCoreOptimizers(self.config)

        # ===== tff core logic =====
        # based on https://colab.research.google.com/github/tensorflow/federated/blob/v0.72.0/docs/tutorials/custom_federated_algorithm_with_tff_optimizers.ipynb

        # local traininig on the federated worker
        @tf.function
        def client_update(model, data, server_weights, client_optimizer):
            # initialize the local model with the current server weights
            client_weights = model.trainable_variables
            tf.nest.map_structure(lambda cw, sw: cw.assign(sw),
                client_weights, server_weights)
            # initialize the local optimizer
            trainable_tensor_specs = tf.nest.map_structure(
                lambda v: tf.TensorSpec(v.shape, v.dtype), client_weights)
            optimizer_state = client_optimizer.initialize(trainable_tensor_specs)
            # update the model locally
            for batch in iter(data):
                with tf.GradientTape() as tape:
                    output = model.forward_pass(batch)
                gradients = tape.gradient(output.loss, client_weights)
                optimizer_state, updated_weights = client_optimizer.next(
                    optimizer_state, client_weights, gradients)
                tf.nest.map_structure(lambda cw, uw: cw.assign(uw),
                    client_weights, updated_weights)
            # return the model deltas to the server for aggregation
            return tf.nest.map_structure(tf.subtract, client_weights, server_weights)

        @attrs.define(eq=False, frozen=True)
        class ServerState(object):
            trainable_weights: Any
            optimizer_state: Any

        @tf.function
        def server_update(server_state, mean_model_delta, server_optimizer):
            # use the aggregated negative model deltas as pseudo gradients
            negative_weights_delta = tf.nest.map_structure(
                lambda w: -1.0 * w, mean_model_delta)
            updated_optimizer_state, updated_weights = server_optimizer.next(
                server_state.optimizer_state, server_state.trainable_weights,
                negative_weights_delta)
            return tff.structure.update_struct(
                server_state,
                trainable_weights=updated_weights,
                optimizer_state=updated_optimizer_state)

        @tff.tf_computation
        def server_init():
            model = cfm()
            trainable_tensor_specs = tf.nest.map_structure(
                lambda v: tf.TensorSpec(v.shape, v.dtype), model.trainable_variables)
            optimizer_state = server_optimizer.initialize(trainable_tensor_specs)
            return ServerState(trainable_weights=model.trainable_variables,
                optimizer_state=optimizer_state)

        @tff.federated_computation
        def server_init_tff():
            return tff.federated_value(server_init(), tff.SERVER)

        server_state_t = server_init.type_signature.result
        trainable_weights_t = server_state_t.trainable_weights

        @tff.tf_computation(server_state_t, trainable_weights_t)
        def server_update_fn(server_state, mean_model_delta):
            return server_update(server_state, mean_model_delta, server_optimizer)

        tf_data_t = tff.SequenceType(tff.types.tensorflow_to_type(cfm().input_spec))

        @tff.tf_computation(tf_data_t, trainable_weights_t)
        def client_update_fn(data, server_weights):
            model = cfm()
            return client_update(model, data, server_weights, client_optimizer)

        federated_server_t = tff.FederatedType(server_state_t, tff.SERVER)
        federated_data_t = tff.FederatedType(tf_data_t, tff.CLIENTS)

        @tff.federated_computation(federated_server_t, federated_data_t)
        def run_one_round(server_state, federated_data):
            # broadcast the aggregated weights to the federated workers
            server_weights_at_client = tff.federated_broadcast(
                server_state.trainable_weights)
            # perform local training and update on the federated workers
            model_deltas = tff.federated_map(
                client_update_fn, (federated_data, server_weights_at_client))
            # centralized aggregation of model deltas from the federated workers
            mean_model_delta = tff.federated_mean(model_deltas)
            # update the centralized model with the averaged model deltas
            server_state = tff.federated_map(
                server_update_fn, (server_state, mean_model_delta))
            return server_state

        # ===== processing =====
        fedavg_process = tff.templates.IterativeProcess(
            initialize_fn=server_init_tff, next_fn=run_one_round)

        self.state = fedavg_process.initialize()
        train_eval = dict()
        for round_idx in range(self.config["num_train_rounds"]):
            self.state = fedavg_process.next(self.state, fed_dataset.train)
            # TODO: evaluate decentralized on train data and log the result for tensorboard
            if(self.config["log_level"] <= logging.DEBUG):
                train_eval[f'{round_idx}'] = self.evaluateDecentralized(fed_dataset.train)
        self.logger.debug(Utils.printEvaluations(train_eval, self.config, first_col_name="Round"))

    @classmethod
    def getTrainedKerasModel(self_class, data, state, config):
        keras_model = KerasModel.createKerasModel(data, config)
        keras_model.compile(loss = getLoss(config),
            metrics = getMetrics(config))
        # assign our weights to the keras model
        tf.nest.map_structure(
            lambda keras_weight, server_weight: keras_weight.assign(server_weight),
            keras_model.trainable_weights, state.trainable_weights)
        return keras_model

    def predict(self, data):
        keras_model = self.getTrainedKerasModel(data, self.state, self.config)
        predictions = KerasModel.predictKerasModel(keras_model, data)
        return predictions

    def evaluate(self, data):
        evaluation_metrics = self.evaluateDecentralized(data)
        self.logger.info(f'Evaluation resulted in {evaluation_metrics}')
        return evaluation_metrics

    def evaluateDecentralized(self, fed_data):
        server_optimizer, client_optimizer = getFedCoreOptimizers(self.config)

        def cfm():
            keras_model = KerasModel.createKerasModel(fed_data[0], self.config)
            fed_model = tff.learning.models.from_keras_model(
                keras_model,
                input_spec = fed_data[0].element_spec,
                loss = getLoss(self.config),
                metrics = getMetrics(self.config))
            return fed_model

        def gm():
            metrics = getMetrics(self.config)
            return metrics

        @tf.function
        def client_evaluate(fed_model, metric_objects, data, server_weights):
            # initialize the model with the trained weights
            client_weights = fed_model.trainable_variables
            tf.nest.map_structure(lambda cw, sw: cw.assign(sw),
                client_weights, server_weights)

            for img_batch, lab_batch in iter(data):
                predictions = fed_model.predict_on_batch(img_batch, training=False)

                # iterate the metrics and update their state w/ current results
                for mtrc in metric_objects:
                    mtrc.update_state(lab_batch, predictions)

            total_metrics = tf.stack([mtrc.result() for mtrc in metric_objects])
            return total_metrics

        @attrs.define(eq=False, frozen=True)
        class ServerState(object):
            trainable_weights: Any
            optimizer_state: Any

        @tff.tf_computation
        def server_init():
            model = cfm()
            trainable_tensor_specs = tf.nest.map_structure(
                lambda v: tf.TensorSpec(v.shape, v.dtype), model.trainable_variables)
            optimizer_state = server_optimizer.initialize(trainable_tensor_specs)
            return ServerState(trainable_weights=model.trainable_variables,
                optimizer_state=optimizer_state)

        @tff.federated_computation
        def server_init_tff():
            return tff.federated_value(server_init(), tff.SERVER)

        server_state_t = server_init.type_signature.result
        trainable_weights_t = server_state_t.trainable_weights

        tf_data_t = tff.SequenceType(tff.types.tensorflow_to_type(cfm().input_spec))

        @tff.tf_computation(tf_data_t, trainable_weights_t)
        def client_evaluate_fn(data, server_weights):
            fed_model = cfm()
            metric_objects = gm()
            evaluation_metrics = client_evaluate(fed_model, metric_objects, data, server_weights)
            return evaluation_metrics

        federated_server_state_t = tff.FederatedType(server_state_t, tff.SERVER)
        federated_data_t = tff.FederatedType(tf_data_t, tff.CLIENTS)

        @tff.federated_computation(federated_server_state_t, federated_data_t)
        def performEvaluation(server_state, federated_data):
            # broadcast the weights to the federated workers
            server_weights_at_client = tff.federated_broadcast(
                server_state.trainable_weights)
            # perform local evaluation on the federated workers with the trained model
            evaluation_metrics_client = tff.federated_map(
                client_evaluate_fn, (federated_data, server_weights_at_client))
            # aggregate the evaluation metrics from the individual federated workers
            evaluation_metrics_agg = tff.federated_mean(evaluation_metrics_client)
            evaluation_output = tff.templates.MeasuredProcessOutput(
                state=server_state,
                result=0,
                measurements=evaluation_metrics_agg)
            return evaluation_output

        eval_process = tff.templates.MeasuredProcess(initialize_fn=server_init_tff,
            next_fn=performEvaluation,
            next_is_multi_arg=True)

        eval_process.initialize()
        evaluation_metrics = eval_process.next(self.state, fed_data).measurements
        metrics = gm()
        evaluation_metrics = {metrics[idx].name: em for idx, em in enumerate(evaluation_metrics)}
        return evaluation_metrics



    def evaluateCentralized(self, data):
        keras_model = self.getTrainedKerasModel(data, self.state, self.config)
        evaluation_metrics = KerasModel.evaluateKerasModel(keras_model, data)
        return evaluation_metrics
