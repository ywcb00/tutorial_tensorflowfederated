import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt
import sys
import getopt
import os
import csv
from enum import Enum
import random

class PartitioningScheme(Enum):
    RANGE       = 1
    RANDOM      = 2
    ROUND_ROBIN = 3

config = {
    "seed": 13,

    "force_load": False,

    "train_response_path": "./data/train_response.csv",
    "val_response_path": "./data/val_response.csv",
    "test_response_path": "./data/test_response.csv",

    "flooded_classes": tf.constant([
            1, # Building-flooded
            3, # Road-flooded
            5, # Water
            # 8, # Pool
        ], dtype=tf.uint32),
    "flooded_threshold": 1/4,

    "part_scheme": PartitioningScheme.ROUND_ROBIN,
    "num_workers": 4,
    "batch_size": 6,

    "num_train_rounds": 5,

    "log_dir": "./log/training",
}

# ===== data loading =====
def getDataset(config):
    train_dir = "./data/FloodNet-Supervised_v1.0/train/train-org-img/"
    val_dir = "./data/FloodNet-Supervised_v1.0/val/val-org-img/"
    test_dir = "./data/FloodNet-Supervised_v1.0/test/test-org-img/"
    # load the image dataset
    train = tf.keras.utils.image_dataset_from_directory(
        directory = train_dir,
        label_mode = None,
        batch_size = None,
        image_size = (3000, 3000),
        shuffle = False,
        # seed = 13,
        color_mode = "rgb")
    val = tf.keras.utils.image_dataset_from_directory(
        directory = val_dir,
        label_mode = None,
        batch_size = None,
        image_size = (3000, 3000),
        shuffle = False,
        # seed = 13,
        color_mode = "rgb")
    test = tf.keras.utils.image_dataset_from_directory(
        directory = test_dir,
        label_mode = None,
        batch_size = None,
        image_size = (3000, 3000),
        shuffle = False,
        # seed = 13,
        color_mode = "rgb")

    # obtain the filenames
    train_fnames = map(lambda fp: os.path.splitext(os.path.basename(fp))[0],
        train.file_paths)
    val_fnames = map(lambda fp: os.path.splitext(os.path.basename(fp))[0],
        val.file_paths)
    test_fnames = map(lambda fp: os.path.splitext(os.path.basename(fp))[0],
        test.file_paths)

    # read/compute the reponse
    train_response = dict()
    val_response = dict()
    test_response = dict()
    if(not config["force_load"] and os.path.exists(config["train_response_path"])
        and os.path.exists(config["val_response_path"]) and os.path.exists(config["test_response_path"])):
        train_response, val_response, test_response = readResponse(config)
    else:
        train_response, val_response, test_response = loadResponse(config)

    # create list of responses in the same order as the image dataset
    train_labels = [[train_response[fname]] for fname in train_fnames]
    val_labels = [[val_response[fname]] for fname in val_fnames]
    test_labels = [[test_response[fname]] for fname in test_fnames]

    # add the responseS to the dataset
    train = tf.data.Dataset.zip((train, tf.data.Dataset.from_tensor_slices(tf.constant(train_labels))))
    val = tf.data.Dataset.zip((val, tf.data.Dataset.from_tensor_slices(tf.constant(val_labels))))
    test = tf.data.Dataset.zip((test, tf.data.Dataset.from_tensor_slices(tf.constant(test_labels))))

    train = train.take(250)
    val = val.take(40)
    test = test.take(40)

    print(f'Found {train.cardinality().numpy()} train instances, {val.cardinality().numpy()} '
        + f'validation instances, and {test.cardinality().numpy()} test instances.')

    return train, val, test

def loadResponse(config):
    def isFlooded(img):
        counts = tf.map_fn(
            fn = lambda fc: tf.reduce_sum(tf.cast(tf.equal(tf.cast(
                tf.round(tf.reshape(img, [-1])), tf.uint32), fc), tf.uint32)),
            elems = config["flooded_classes"])
        flooded_count = tf.reduce_sum(counts)
        # return True if the count of flooded pixels exceeds the specified threshold
        return tf.greater(flooded_count, tf.cast(tf.multiply(tf.cast(tf.size(img), tf.float32),
                config["flooded_threshold"]), tf.uint32))

    # read the response label images from disk
    train_labels_dir = "./data/FloodNet-Supervised_v1.0/train/train-label-img/"
    val_labels_dir = "./data/FloodNet-Supervised_v1.0/val/val-label-img/"
    test_labels_dir = "./data/FloodNet-Supervised_v1.0/test/test-label-img/"
    train_labels = tf.keras.utils.image_dataset_from_directory(
        directory = train_labels_dir,
        label_mode = None,
        batch_size = None,
        image_size = (3000, 3000),
        shuffle = False,
        # seed = 13,
        color_mode = "grayscale")
    val_labels = tf.keras.utils.image_dataset_from_directory(
        directory = val_labels_dir,
        label_mode = None,
        batch_size = None,
        image_size = (3000, 3000),
        shuffle = False,
        # seed = 13,
        color_mode = "grayscale")
    test_labels = tf.keras.utils.image_dataset_from_directory(
        directory = test_labels_dir,
        label_mode = None,
        batch_size = None,
        image_size = (3000, 3000),
        shuffle = False,
        # seed = 13,
        color_mode = "grayscale")

    train_fnames = map(lambda fp: os.path.basename(fp).split("_")[0],
        train_labels.file_paths)
    val_fnames = map(lambda fp: os.path.basename(fp).split("_")[0],
        val_labels.file_paths)
    test_fnames = map(lambda fp: os.path.basename(fp).split("_")[0],
        test_labels.file_paths)

    # create the response (True/False) from the label images
    train_response = train_labels.map(isFlooded)
    val_response = val_labels.map(isFlooded)
    test_response = test_labels.map(isFlooded)

    train_response = map(lambda tb: tb.numpy(), train_labels.map(isFlooded))
    val_response = map(lambda tb: tb.numpy(), val_labels.map(isFlooded))
    test_response = map(lambda tb: tb.numpy(), test_labels.map(isFlooded))

    train_response = dict(zip(train_fnames, train_response))
    val_response = dict(zip(val_fnames, val_response))
    test_response = dict(zip(test_fnames, test_response))

    # write the computed binary responses to disk for later usage
    with open(config["train_response_path"], "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["fname", "response"])
        for key, val in train_response.items():
            writer.writerow([key, val])
    with open(config["val_response_path"], "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["fname", "response"])
        for key, val in val_response.items():
            writer.writerow([key, val])
    with open(config["test_response_path"], "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["fname", "response"])
        for key, val in test_response.items():
            writer.writerow([key, val])

    print(f'Wrote {len(train_response)} training labels, {len(val_response)} '
        + f'validation labels, and {len(test_response)} test labels to disk.')

    return train_response, val_response, test_response

def readResponse(config):
    train_response = dict()
    with open(config["train_response_path"], "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # omit the header
        for row in reader:
            train_response[row[0]] = (row[1] == "True")
    val_response = dict()
    with open(config["val_response_path"], "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # omit the header
        for row in reader:
            val_response[row[0]] = (row[1] == "True")
    test_response = dict()
    with open(config["test_response_path"], "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # omit the header
        for row in reader:
            test_response[row[0]] = (row[1] == "True")

    print(f'Read {len(train_response)} training labels, {len(val_response)} '
        + f'validation labels, and {len(test_response)} test labels from disk.')

    return train_response, val_response, test_response

# ===== partitioning =====
def partitionTrainData(train, config):
    n_workers = config["num_workers"]
    match config["part_scheme"]:
        case PartitioningScheme.RANGE:
            train_parts = partitionTrainDataRange(train, n_workers)
            return train_parts
        case PartitioningScheme.RANDOM:
            train.shuffle(train.cardinality(), seed=config["seed"])
            train_parts = partitionTrainDataRange(train, n_workers)
            return train_parts
        case PartitioningScheme.ROUND_ROBIN:
            train_parts = [train.shard(n_workers, w_idx) for w_idx in range(n_workers)]
            return train_parts

def partitionTrainDataRange(train, n_workers):
    n_rows = train.cardinality().numpy()
    distribute_remainder = lambda idx: 1 if idx < (n_rows % n_workers) else 0
    train_parts = list()
    num_elements = 0
    for w_idx in range(n_workers):
        train.skip(num_elements)
        num_elements = (n_rows // n_workers) + distribute_remainder(w_idx)
        train_parts.append(train.take(num_elements))
    return train_parts

# ===== model creation =====
def createKerasModel(train, config):
    # load the pre-defined ResNet50 model with 2 output classes and not pre-trained
    # model = tf.keras.applications.resnet50.ResNet50(
    #     include_top = True,
    #     weights = None,
    #     input_shape = train.element_spec[0].shape[1:],
    #     classes = 2)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=train.element_spec[0].shape[1:]),
        tf.keras.layers.Conv2D(3, 3, strides=2, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
    ])

    return model

def createFedModel(train, config):
    model = createKerasModel(train, config)

    fed_model = tff.learning.models.from_keras_model(
        keras_model = model,
        input_spec = train.element_spec,
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics = [tf.keras.metrics.BinaryCrossentropy()])

    return fed_model

# ===== model training =====
def trainFedModel(train, fed_train, config):
    def cfm():
        return createFedModel(train, config)

    training_process = tff.learning.algorithms.build_weighted_fed_avg(cfm,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

    # set logging for tensorboard visualization
    logdir = config["log_dir"] # delete any previous results
    try:
        tf.io.gfile.rmtree(logdir)
    except tf.errors.NotFoundError as e:
        pass # ignore if no previous results to delete
    log_summary_writer = tf.summary.create_file_writer(logdir)

    training_state = training_process.initialize()

    with log_summary_writer.as_default():
        for n_round in range(config["num_train_rounds"]):
            result = training_process.next(training_state, fed_train)
            training_state = result.state
            training_metrics = result.metrics

            for name, value in training_metrics['client_work']['train'].items():
                tf.summary.scalar(name, value, step=n_round)

            print(f'Training round {n_round}: {training_metrics}')

def main(argv):
    try:
        opts, args = getopt.getopt(argv[1:], "hl", ["help", "forceload"])
    except getopt.GetoptError:
        print("Wrong usage.")
        print("Usage:", argv[0], "[--forceload]")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("Usage:", argv[0], "[--forceload]")
            sys.exit()
        elif opt in ("-l", "--forceload"):
            config["force_load"] = True

    # obtain the dataset (either load or compute the response labels)
    train, val, test = getDataset(config)

    # preprocess the dataset for federated training
    fed_train = partitionTrainData(train, config)
    train = train.batch(config["batch_size"])
    fed_train = [ft.batch(config["batch_size"]) for ft in fed_train]

    # create and train the model
    trainFedModel(train, fed_train, config)


if __name__ == '__main__':
    main(sys.argv)
