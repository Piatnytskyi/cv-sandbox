import numpy as np
from .initialization_base import InitializationStrategy

class ForgyInitialization(InitializationStrategy):
    def initialize(self, data: np.ndarray, n_clusters: int, random_state: int = None) -> np.ndarray:
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples = data.shape[0]
        indices = np.random.choice(n_samples, size=n_clusters, replace=False)
        return data[indices].copy()
