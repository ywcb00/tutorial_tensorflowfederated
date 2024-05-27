from abc import ABC, abstractmethod

class IModel(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def evaluate(self, data):
        pass
