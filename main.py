from IDataset import DatasetID
from FedDataset import FedDataset, PartitioningScheme
from FedKerasModel import FedKerasModel
from KerasModel import KerasModel
from Utils import Utils

import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt
import sys
import getopt
import os
import csv

config = {
    "seed": 13,

    "force_load": False,

    "dataset_id": DatasetID.FloodNet,

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
    "num_workers": 2,
    "batch_size": 2,

    "model": "c64_c32_avg_dr50_dr25",
    "num_train_rounds": 3,

    "log_dir": "./log/training",
}

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
    dataset = Utils.getDataset(config)
    dataset.load()

    # construct data partitions for federated execution
    fed_dataset = FedDataset(config)
    fed_dataset.construct(dataset)
    fed_dataset.batch()

    dataset.batch()

    # ===== Local Training =====
    # create and fit the local keras model
    keras_model = KerasModel(config)
    keras_model.fit(dataset)

    print(keras_model.predict(dataset.train))

    # evaluate the model
    evaluation_metrics = keras_model.evaluate(dataset.val)
    print(evaluation_metrics)


    # ===== Federated Training =====
    # create and fit the federated model
    fed_keras_model = FedKerasModel(config)
    fed_keras_model.fit(fed_dataset)

    print(fed_keras_model.predict(dataset.train))

    # evaluate the model
    evaluation_metrics = fed_keras_model.evaluate(fed_dataset.val)
    print(evaluation_metrics)
    evaluation_metrics = fed_keras_model.evaluateCentralized(dataset.val)
    print(evaluation_metrics)


if __name__ == '__main__':
    main(sys.argv)
