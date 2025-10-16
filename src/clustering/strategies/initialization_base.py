from abc import ABC, abstractmethod
import numpy as np

class InitializationStrategy(ABC):
    @abstractmethod
    def initialize(self, data: np.ndarray, n_clusters: int, random_state: int = None) -> np.ndarray:
        pass
