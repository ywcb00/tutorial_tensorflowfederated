from abc import ABC, abstractmethod
from enum import Enum

class DatasetID(Enum):
    FloodNet = 1

class IDataset(ABC):
    def __init__(self, config):
        self.config = config
        self.train = None
        self.val = None
        self.test = None

    @abstractmethod
    def load(self):
        pass

    def batch(self):
        self.train = self.train.batch(self.config["batch_size"])
        self.val = self.val.batch(self.config["batch_size"])
        self.test = self.test.batch(self.config["batch_size"])
