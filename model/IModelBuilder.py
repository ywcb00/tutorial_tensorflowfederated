from abc import ABC, abstractmethod

class IModelBuilder(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def buildModel(self):
        pass

    @abstractmethod
    def getLoss(self):
        pass

    @abstractmethod
    def getMetrics(self):
        pass

    @abstractmethod
    def getLearningRate(self):
        pass

    @abstractmethod
    def getOptimizer(self):
        pass
