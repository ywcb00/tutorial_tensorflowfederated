from dataset.IDataset import DatasetID
from dataset.FedDataset import FedDataset, PartitioningScheme
from model.FedKerasModel import FedKerasModel
from model.KerasModel import KerasModel
from utils.Utils import Utils

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
    "batch_size": 6,

    "model": "c96_c32_dr25",
    "num_train_rounds": 10,
    "learning_rate": 0.00005,

    "log_dir": "./log/training",
}

def trainLocalKeras(dataset, config):
    # ===== Local Training =====
    # create and fit the local keras model
    keras_model = KerasModel(config)
    keras_model.fit(dataset)

    print(keras_model.predict(dataset.train)[1:10])
    print(list(map(lambda e: e[1], list(dataset.train.as_numpy_iterator())[0:2])))

    print(keras_model.predict(dataset.val)[1:10])
    print(list(map(lambda e: e[1], list(dataset.val.as_numpy_iterator())[0:2])))

    # evaluate the model
    evaluation_metrics = keras_model.evaluate(dataset.val)
    print(evaluation_metrics)
    return evaluation_metrics

def trainFedKeras(dataset, fed_dataset, config):
    # ===== Federated Training =====
    # create and fit the federated model
    fed_keras_model = FedKerasModel(config)
    fed_keras_model.fit(fed_dataset)

    print(fed_keras_model.predict(dataset.train)[1:10])

    # evaluate the model
    evaluation_metrics = fed_keras_model.evaluate(fed_dataset.val)
    print(evaluation_metrics)
    evaluation_metrics = fed_keras_model.evaluateCentralized(dataset.val)
    print(evaluation_metrics)

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

    # trainFedKeras(dataset, fed_dataset, config)
    trainLocalKeras(dataset, config)

    # model_abbrvs = [
    #     "c10_avg_dr25",
    #     "c20_c10_avg_dr25",
    #     "c40_c20_c10_avg_dr25",
    #     "c32_c64_c16_avg_fl_dr50",
    #     "c64_c32_avg_dr50_dr25",
    #     "c64_c32_c32_avg_dr50_dr25",
    #     "c96_c64_c32_avg_dr50_dr25"
    # ]

    # model_evaluations = dict()

    # for ma in model_abbrvs:
    #     config["model"] = ma
    #     model_evaluations[ma] = trainLocalKeras(dataset, config)

    # print(model_evaluations)

    # learning_rates = [
    #     0.1,
    #     0.01,
    #     0.001,
    #     0.0005,
    #     0.0001,
    #     0.00005,
    #     0.00001
    # ]

    # model_evaluations = dict()

    # for lr in learning_rates:
    #     print(f'Training with learning rate {lr}')
    #     config["learning_rate"] = lr
    #     model_evaluations[lr] = trainLocalKeras(dataset, config)

    # print(model_evaluations)



if __name__ == '__main__':
    main(sys.argv)
