from IModelBuilder import IModelBuilder

import tensorflow as tf

class FloodNetModelBuilder(IModelBuilder):
    def buildModel(self, data):
        # load the pre-defined ResNet50 model with 2 output classes and not pre-trained
        # model = tf.keras.applications.resnet50.ResNet50(
        #     include_top = True,
        #     weights = None,
        #     input_shape = data.element_spec[0].shape[1:],
        #     classes = 1)

        # construct a sequential model
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=data.element_spec[0].shape[1:]))
        self.buildKerasModelLayers(model)

        return model

    def buildKerasModelLayers(self, keras_model):
        # NOTE: set the initializers in order to ensure reproducibility
        match self.config["model"]:
            case "c10_avg_dr25":
                # first layer is the input
                keras_model.add(tf.keras.layers.Conv2D(10, 10, strides=5, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.BatchNormalization(
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones(),
                    moving_mean_initializer=tf.keras.initializers.Zeros(),
                    moving_variance_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.GlobalAveragePooling2D())
                keras_model.add(tf.keras.layers.Dropout(0.25, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
            case "c20_c10_avg_dr25":
                # first layer is the input
                keras_model.add(tf.keras.layers.Conv2D(20, 10, strides=10, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.BatchNormalization(
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones(),
                    moving_mean_initializer=tf.keras.initializers.Zeros(),
                    moving_variance_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.Conv2D(10, 8, strides=10, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.BatchNormalization(
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones(),
                    moving_mean_initializer=tf.keras.initializers.Zeros(),
                    moving_variance_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.GlobalAveragePooling2D())
                keras_model.add(tf.keras.layers.Dropout(0.25, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
            case "c40_c20_c10_avg_dr25":
                # first layer is the input
                keras_model.add(tf.keras.layers.Conv2D(40, 10, strides=5, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.BatchNormalization(
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones(),
                    moving_mean_initializer=tf.keras.initializers.Zeros(),
                    moving_variance_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.Conv2D(20, 10, strides=10, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.BatchNormalization(
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones(),
                    moving_mean_initializer=tf.keras.initializers.Zeros(),
                    moving_variance_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.Conv2D(10, 8, strides=10, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.BatchNormalization(
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones(),
                    moving_mean_initializer=tf.keras.initializers.Zeros(),
                    moving_variance_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.GlobalAveragePooling2D())
                keras_model.add(tf.keras.layers.Dropout(0.25, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
            case "c32_c64_c16_avg_fl_dr50":
                # first layer is the input
                keras_model.add(tf.keras.layers.Conv2D(32, (11, 11), strides=2, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.BatchNormalization(
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones(),
                    moving_mean_initializer=tf.keras.initializers.Zeros(),
                    moving_variance_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.Conv2D(64, (11, 11), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.BatchNormalization(
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones(),
                    moving_mean_initializer=tf.keras.initializers.Zeros(),
                    moving_variance_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.Conv2D(16, (11, 11), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.BatchNormalization(
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones(),
                    moving_mean_initializer=tf.keras.initializers.Zeros(),
                    moving_variance_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.GlobalAveragePooling2D())
                keras_model.add(tf.keras.layers.Dropout(0.5, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Flatten())
                keras_model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.sigmoid,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
            case "c64_c32_avg_dr50_dr25":
                # first layer is the input
                keras_model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.GroupNormalization(groups=16,
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.Conv2D(32, (5, 5), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.GroupNormalization(groups=16,
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.GlobalAveragePooling2D())
                keras_model.add(tf.keras.layers.Dropout(0.5, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.Dropout(0.25, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
            case "c64_c32_c32_avg_dr50_dr25":
                # first layer is the input
                keras_model.add(tf.keras.layers.Conv2D(64, (11, 11), strides=7, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.GroupNormalization(groups=16,
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.Conv2D(32, (7, 7), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.GroupNormalization(groups=16,
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.Conv2D(32, (5, 5), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.GroupNormalization(groups=16,
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.GlobalAveragePooling2D())
                keras_model.add(tf.keras.layers.Dropout(0.5, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.Dropout(0.25, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
            case "c96_c64_c32_avg_dr50_dr25":
                # first layer is the input
                keras_model.add(tf.keras.layers.Conv2D(96, (11, 11), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.GroupNormalization(groups=16,
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.Conv2D(64, (11, 11), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.GroupNormalization(groups=16,
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.Conv2D(32, (7, 7), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.GroupNormalization(groups=16,
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.GlobalAveragePooling2D())
                keras_model.add(tf.keras.layers.Dropout(0.5, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.Dropout(0.25, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
            case "c96_c32_dr25_dr50":
                # first layer is the input
                keras_model.add(tf.keras.layers.Conv2D(96, (17, 17), strides=5, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.GroupNormalization(groups=16,
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((5, 5)))
                keras_model.add(tf.keras.layers.Conv2D(32, (11, 11), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.GroupNormalization(groups=8,
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.Dropout(0.25, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Flatten())
                keras_model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.Dropout(0.5, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
            case "c96_c32_dr25":
                # first layer is the input
                keras_model.add(tf.keras.layers.Conv2D(96, (17, 17), strides=5, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros(),
                    activation=tf.keras.activations.relu))
                keras_model.add(tf.keras.layers.MaxPool2D((5, 5)))
                keras_model.add(tf.keras.layers.Conv2D(32, (5, 5), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros(),
                    activation=tf.keras.activations.relu))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.Dropout(0.25, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Flatten())
                keras_model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
