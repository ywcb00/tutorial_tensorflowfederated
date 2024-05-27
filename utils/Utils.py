from dataset.IDataset import DatasetID
from dataset.FloodNetDataset import FloodNetDataset
from model.FloodNetModelBuilder import FloodNetModelBuilder

class Utils:
    @classmethod
    def getDataset(self_class, config):
        match config["dataset_id"]:
            case DatasetID.FloodNet:
                return FloodNetDataset(config)

    @classmethod
    def getModelBuilder(self_class, config):
        match config["dataset_id"]:
            case DatasetID.FloodNet:
                return FloodNetModelBuilder(config)
