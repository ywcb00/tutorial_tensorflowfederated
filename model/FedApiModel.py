from model.IModel import IModel
from model.KerasModel import KerasModel
from model.ModelBuilderUtils import getLoss, getMetrics, getFedApiOptimizers

import logging
import tensorflow as tf
import tensorflow_federated as tff

class FedApiModel(IModel):
    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger("model/FedApiModel")
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
            return self.createFedModel(fed_dataset.train, self.config)

        training_process = tff.learning.algorithms.build_weighted_fed_avg(cfm,
            server_optimizer_fn=lambda: getFedApiOptimizers(self.config)[0],
            client_optimizer_fn=lambda: getFedApiOptimizers(self.config)[1])

        # set logging for tensorboard visualization
        logdir = self.config["log_dir"] # delete any previous results
        try:
            tf.io.gfile.rmtree(logdir)
        except tf.errors.NotFoundError as e:
            pass # ignore if no previous results to delete
        log_summary_writer = tf.summary.create_file_writer(logdir)
        log_summary_writer.set_as_default()

        training_state = training_process.initialize()

        for n_round in range(self.config["num_train_rounds"]):
            training_result = training_process.next(training_state, fed_dataset.train)
            training_state = training_result.state
            training_metrics = training_result.metrics

            for name, value in training_metrics['client_work']['train'].items():
                tf.summary.scalar(name, value, step=n_round)

            self.logger.debug(f'Training round {n_round}: {training_metrics}')

        self.state = (training_process, training_result)

    def predict(self, data):
        keras_model = self.getTrainedKerasModel(data, self.state, self.config)
        predictions = KerasModel.predictKerasModel(keras_model, data)
        return predictions

    @classmethod
    def getTrainedKerasModel(self_class, data, state, config):
        keras_model = KerasModel.createKerasModel(data, config)
        keras_model.compile(loss = getLoss(config),
            metrics = getMetrics(config))
        model_weights = state[0].get_model_weights(state[1].state)
        model_weights.assign_weights_to(keras_model)
        return keras_model

    def evaluate(self, data):
        evaluation_metrics = self.evaluateDecentralized(data)
        self.logger.info(f'Evaluation resulted in {evaluation_metrics}')
        return evaluation_metrics

    def evaluateDecentralized(self, fed_data):
        def cfm():
            return self.createFedModel(fed_data, self.config)

        evaluation_process = tff.learning.build_federated_evaluation(cfm)
        model_weights = self.state[0].get_model_weights(self.state[1].state)
        evaluation_metrics = evaluation_process(model_weights, fed_data)['eval']
        return evaluation_metrics

    def evaluateCentralized(self, data):
        keras_model = self.getTrainedKerasModel(data, self.state, self.config)
        evaluation_metrics = KerasModel.evaluateKerasModel(keras_model, data)
        return evaluation_metrics
