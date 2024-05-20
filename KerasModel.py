from IModel import IModel
from Utils import Utils

import tensorflow as tf

class KerasModel(IModel):
    @classmethod
    def createKerasModel(self_class, data, config):
        model_builder = Utils.getModelBuilder(config)
        keras_model = model_builder.buildModel(data)
        return keras_model

    def fit(self, dataset):
        self.model = self.createKerasModel(dataset.train, self.config)
        self.model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=0.02),
            loss = tf.keras.losses.BinaryCrossentropy(),
            metrics = [tf.keras.metrics.BinaryCrossentropy()])
        self.fit_history = self.model.fit(x=dataset.train,
            y=None, # already in the dataset
            batch_size=None, # already in the dataset
            epochs=1,
            validation_data=None, # we have a separate validation split
            shuffle=False)

    def predict(self, data):
        return self.predictKerasModel(self.model, data)

    @classmethod
    def predictKerasModel(self_class, keras_model, data):
        predictions = keras_model.predict(data)
        return predictions

    def evaluate(self, data):
        return self.evaluateKerasModel(self.model, data)

    @classmethod
    def evaluateKerasModel(self_class, keras_model, data):
        evaluation_scalars = keras_model.evaluate(data)
        evaluation_metrics = dict(zip(keras_model.metrics_names, evaluation_scalars))
        return evaluation_metrics
