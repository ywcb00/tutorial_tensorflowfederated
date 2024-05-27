from model.IModel import IModel
from model.KerasModel import KerasModel
from utils.Utils import Utils

import tensorflow as tf
import tensorflow_federated as tff

class FedKerasModel(IModel):
    @classmethod
    def createFedModel(self_class, fed_data, config):
        keras_model = KerasModel.createKerasModel(fed_data[0], config)
        fed_model = tff.learning.models.from_keras_model(
            keras_model = keras_model,
            input_spec = fed_data[0].element_spec,
            loss = tf.keras.losses.BinaryCrossentropy(),
            metrics = [tf.keras.metrics.BinaryCrossentropy(),
                tf.keras.metrics.BinaryAccuracy()])
        return fed_model

    def fit(self, fed_dataset):
        def cfm():
            return self.createFedModel(fed_dataset.train, self.config)

        training_process = tff.learning.algorithms.build_weighted_fed_avg(cfm,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

        # set logging for tensorboard visualization
        logdir = self.config["log_dir"] # delete any previous results
        try:
            tf.io.gfile.rmtree(logdir)
        except tf.errors.NotFoundError as e:
            pass # ignore if no previous results to delete
        log_summary_writer = tf.summary.create_file_writer(logdir)

        training_state = training_process.initialize()

        with log_summary_writer.as_default():
            for n_round in range(self.config["num_train_rounds"]):
                training_result = training_process.next(training_state, fed_dataset.train)
                training_state = training_result.state
                training_metrics = training_result.metrics

                for name, value in training_metrics['client_work']['train'].items():
                    tf.summary.scalar(name, value, step=n_round)

                print(f'Training round {n_round}: {training_metrics}')

        self.state = (training_process, training_result)

    def predict(self, data):
        keras_model = self.getTrainedKerasModel(data, self.state, self.config)
        predictions = KerasModel.predictKerasModel(keras_model, data)
        return predictions

    @classmethod
    def getTrainedKerasModel(self, data, state, config):
        keras_model = KerasModel.createKerasModel(data, config)
        keras_model.compile(loss = tf.keras.losses.BinaryCrossentropy(),
            metrics = [tf.keras.metrics.BinaryCrossentropy(),
                tf.keras.metrics.BinaryAccuracy()])
        model_weights = state[0].get_model_weights(state[1].state)
        model_weights.assign_weights_to(keras_model)
        return keras_model

    def evaluate(self, data):
        return self.evaluateDecentralized(data)

    def evaluateDecentralized(self, fed_data):
        def cfm():
            return self.createFedModel(fed_data, self.config)

        evaluation_process = tff.learning.build_federated_evaluation(cfm)
        model_weights = self.state[0].get_model_weights(self.state[1].state)
        evaluation_metrics = evaluation_process(model_weights, fed_data)
        return evaluation_metrics

    def evaluateCentralized(self, data):
        keras_model = self.getTrainedKerasModel(data, self.state, self.config)
        evaluation_metrics = KerasModel.evaluateKerasModel(keras_model, data)
        return evaluation_metrics
