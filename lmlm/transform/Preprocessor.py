from abc import ABC, abstractmethod

class Preprocessor(ABC):
    def __init__(self):
        self.mean = None
        self.std = None

    def __call__(self, point, **kwargs):
        return preprocess(point, kwargs)

    @abstractmethod
    def preprocess(self, point, **kwargs):
        pass

    def normalise(self, point, **kwargs):
        if not self.centre.__isabstractmethod__:
            point = self.centre(point)
        if not self.scale.__isabstractmethod__:
            point = self.scale(point)
        return point

    @abstractmethod
    def centre(self, point, **kwargs):
        pass

    @abstractmethod
    def scale(self, point, **kwargs):
        pass

    @abstractmethod
    def fit(self, points, **kwargs):
        pass
