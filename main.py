import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt
import sys
import getopt
import os
import csv

config = {
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
}

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
    train_labels = [train_response[fname] for fname in train_fnames]
    val_labels = [val_response[fname] for fname in val_fnames]
    test_labels = [test_response[fname] for fname in test_fnames]

    # assign the response class to the dataset images
    train = tf.data.Dataset.zip((train, tf.data.Dataset.from_tensor_slices(tf.constant(train_labels))))
    val = tf.data.Dataset.zip((val, tf.data.Dataset.from_tensor_slices(tf.constant(val_labels))))
    test = tf.data.Dataset.zip((test, tf.data.Dataset.from_tensor_slices(tf.constant(test_labels))))

    for images, labels in train.take(10):
        tmp_single_img = tf.divide(images, tf.reduce_max(images))
        print(labels.numpy())
        plt.imshow(tmp_single_img)
        plt.show()

    train = train.take(100)
    val = val.take(40)
    test = test.take(40)

    print(f'Found {train.__len__()} train instances, {val.__len__()} '
        + f'validation instances, and {test.__len__()} test instances.')

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
            train_response[row[0]] = row[1]
    val_response = dict()
    with open(config["val_response_path"], "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # omit the header
        for row in reader:
            val_response[row[0]] = row[1]
    test_response = dict()
    with open(config["test_response_path"], "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # omit the header
        for row in reader:
            test_response[row[0]] = (row[1] == "True")

    print(f'Read {len(train_response)} training labels, {len(val_response)} '
        + f'validation labels, and {len(test_response)} test labels from disk.')

    return train_response, val_response, test_response


def createFedModel(train):
    # load the pre-defined ResNet50 model with 2 output classes and not pre-trained
    model = tf.keras.applications.resnet50.ResNet50(
        include_top = True,
        weights = None,
        input_shape = (3000, 3000, 3),
        classes = 2)

    fed_model = tff.learning.models.from_keras_model(
        keras_model = model,
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        input_spec = train.element_spec,
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])

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

    train, val, test = getDataset(config)
    # createFedModel(train)

if __name__ == '__main__':
    main(sys.argv)
