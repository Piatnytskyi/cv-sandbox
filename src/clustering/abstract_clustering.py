from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

class AbstractClustering(ABC):
    def __init__(self, n_clusters: int, random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        
    @abstractmethod
    def fit(self, data: np.ndarray) -> 'AbstractClustering':
        pass
    
    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        pass
