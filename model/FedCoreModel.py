from model.IModel import IModel
from model.KerasModel import KerasModel
from model.ModelBuilderUtils import getLoss, getMetrics, getFedCoreOptimizers

import attrs
import tensorflow as tf
import tensorflow_federated as tff
from typing import Any

class FedCoreModel(IModel):
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
        for round_idx in range(self.config["num_train_rounds"]):
            self.state = fedavg_process.next(self.state, fed_dataset.train)

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
        return self.evaluateCentralized(data)

    # def evaluateDecentralized(self, fed_data):
    #     def cfm():
    #         return self.createFedModel(fed_data, self.config)

    #     evaluation_process = tff.learning.build_federated_evaluation(cfm)
    #     model_weights = self.state[0].get_model_weights(self.state[1].state)
    #     evaluation_metrics = evaluation_process(model_weights, fed_data)
    #     return evaluation_metrics

    def evaluateCentralized(self, data):
        keras_model = self.getTrainedKerasModel(data, self.state, self.config)
        evaluation_metrics = KerasModel.evaluateKerasModel(keras_model, data)
        return evaluation_metrics
