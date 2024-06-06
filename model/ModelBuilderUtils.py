from dataset.DatasetUtils import DatasetID
from model.FloodNetModelBuilder import FloodNetModelBuilder
from model.MnistModelBuilder import MnistModelBuilder

def getModelBuilder(config):
    match config["dataset_id"]:
        case DatasetID.FloodNet:
            return FloodNetModelBuilder(config)
        case DatasetID.Mnist:
            return MnistModelBuilder(config)

def getLoss(config):
    return getModelBuilder(config).getLoss()

def getMetrics(config):
    return getModelBuilder(config).getMetrics()

def getLearningRate(config):
    return getModelBuilder(config).getLearningRate()

def getOptimizer(config):
    return getModelBuilder(config).getOptimizer()

def getFedLearningRates(config):
    return getModelBuilder(config).getFedLearningRates()

def getFedApiOptimizers(config):
    return getModelBuilder(config).getFedApiOptimizers()

def getFedCoreOptimizers(config):
    return getModelBuilder(config).getFedCoreOptimizers()
