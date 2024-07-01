from tffdataset.DatasetUtils import DatasetID, getDataset
from tffdataset.FedDataset import FedDataset, PartitioningScheme
from tffmodel.FedCoreModel import FedCoreModel
from tffmodel.FedApiModel import FedApiModel
from tffmodel.KerasModel import KerasModel
from tffmodel.ModelUtils import ModelUtils

import getopt
import logging
import sys
import tensorflow as tf

config = {
    "seed": 13,

    "force_load": False,

    "dataset_id": DatasetID.Mnist,

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

    "num_train_rounds": 10,

    "log_dir": "./log/training",
    "log_level": logging.DEBUG,
}

def trainLocalKeras(dataset, config):
    # ===== Local Training =====
    # create and fit the local keras model
    keras_model = KerasModel(config)
    keras_model.fit(dataset)

    # evaluate the model
    evaluation_metrics = keras_model.evaluate(dataset.val)
    return evaluation_metrics

def trainFedApi(dataset, fed_dataset, config):
    # ===== Federated Training =====
    # create and fit the federated model with tff api
    fed_api_model = FedApiModel(config)
    fed_api_model.fit(fed_dataset)

    # evaluate the model
    evaluation_metrics = fed_api_model.evaluate(fed_dataset.val)
    # evaluation_metrics = fed_api_model.evaluateCentralized(dataset.val)
    return evaluation_metrics

def trainFedCore(dataset, fed_dataset, config):
    # ===== Federated Training =====
    # create and fit the federated model with tff core
    fed_core_model = FedCoreModel(config)
    fed_core_model.fit(fed_dataset)

    # evaluate the model
    evaluation_metrics = fed_core_model.evaluate(fed_dataset.val)
    # evaluation_metrics = fed_core_model.evaluateCentralized(dataset.val)
    return evaluation_metrics

def main(argv):
    logger = logging.getLogger("main.py")
    logger.setLevel(config["log_level"])

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
    dataset = getDataset(config)
    dataset.load()

    # construct data partitions for federated execution
    fed_dataset = FedDataset(config)
    fed_dataset.construct(dataset)
    fed_dataset.batch()

    dataset.batch()

    evaluations = dict()
    evaluations["keras"] = trainLocalKeras(dataset, config)
    evaluations["fedapi"] = trainFedApi(dataset, fed_dataset, config)
    evaluations["fedcore"] = trainFedCore(dataset, fed_dataset, config)

    logger.info(ModelUtils.printEvaluations(evaluations, config))

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
