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

    @abstractmethod
    def getFedLearningRates(self):
        pass

    @abstractmethod
    def getFedApiOptimizers(self):
        pass

    @abstractmethod
    def getFedCoreOptimizers(self):
        pass
