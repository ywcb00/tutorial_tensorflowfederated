from model.IModelBuilder import IModelBuilder

import tensorflow as tf

class MnistModelBuilder(IModelBuilder):
    def __init__(self, config):
        super().__init__(config)
        self.learning_rate = 0.001

    def buildModel(self, data):
        # construct a sequential model
        model = tf.keras.Sequential()

        model.add(tf.keras.Input(shape=data.element_spec[0].shape[1:]))
        self.buildKerasModelLayers(model)

        return model

    def buildKerasModelLayers(self, keras_model):
        num_classes = 10

        # NOTE: set the initializers in order to ensure reproducibility
        keras_model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
            bias_initializer=tf.keras.initializers.Zeros()))
        keras_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        keras_model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
            bias_initializer=tf.keras.initializers.Zeros()))
        keras_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        keras_model.add(tf.keras.layers.Flatten())
        keras_model.add(tf.keras.layers.Dropout(0.5, seed=self.config["seed"]))
        keras_model.add(tf.keras.layers.Dense(num_classes, activation="softmax",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
            bias_initializer=tf.keras.initializers.Zeros()))

    def getLoss(self):
        return tf.keras.losses.CategoricalCrossentropy()

    def getMetrics(self):
        return [tf.metrics.CategoricalCrossentropy(),
            tf.metrics.CategoricalAccuracy()]

    def getLearningRate(self):
        return self.learning_rate

    def getOptimizer(self):
        return tf.keras.optimizers.SGD(learning_rate=self.getLearningRate())

