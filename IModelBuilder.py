from abc import ABC, abstractmethod

class IModelBuilder(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def buildModel(self):
        pass
