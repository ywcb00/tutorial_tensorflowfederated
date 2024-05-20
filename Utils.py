from IDataset import DatasetID
from FloodNetDataset import FloodNetDataset
from FloodNetModelBuilder import FloodNetModelBuilder

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
