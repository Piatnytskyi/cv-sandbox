import numpy as np
from typing import Optional, Callable

from .abstract_clustering import AbstractClustering
from .strategies import InitializationStrategy, ForgyInitialization
from ..config.constants import MAX_ITERATIONS, CONVERGENCE_TOLERANCE

class KMeans(AbstractClustering):
    def __init__(
        self,
        n_clusters: int,
        initialization_strategy: Optional[InitializationStrategy] = None,
        max_iterations: int = MAX_ITERATIONS,
        tolerance: float = CONVERGENCE_TOLERANCE,
        random_state: Optional[int] = None,
        progress_callback: Optional[Callable[[int, float], None]] = None
    ):
        super().__init__(n_clusters, random_state, progress_callback)
        self.initialization_strategy = initialization_strategy or ForgyInitialization()
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
    def fit(self, data: np.ndarray) -> 'KMeans':       
        self.cluster_centers_ = self.initialization_strategy.initialize(
            data, self.n_clusters, self.random_state
        )
        
        for iteration in range(self.max_iterations):
            self.labels_ = self._assign_clusters(data)
            new_centers = self._compute_centroids(data)
            center_shift = np.sum((new_centers - self.cluster_centers_) ** 2)
            
            self.cluster_centers_ = new_centers
            self.n_iter_ = iteration + 1
            
            if self.progress_callback:
                self.progress_callback(iteration + 1, center_shift)
            
            if center_shift < self.tolerance:
                break
        
        self.inertia_ = self._compute_inertia(data)
        
        return self
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        return self._assign_clusters(data)
    
    def _assign_clusters(self, data: np.ndarray) -> np.ndarray:
        distances = np.sqrt(
            ((data[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]) ** 2).sum(axis=2)
        )
        
        return np.argmin(distances, axis=1)
    
    def _compute_centroids(self, data: np.ndarray) -> np.ndarray:
        n_features = data.shape[1]
        new_centers = np.zeros((self.n_clusters, n_features))
        
        for i in range(self.n_clusters):
            cluster_points = data[self.labels_ == i]
            if len(cluster_points) > 0:
                new_centers[i] = cluster_points.mean(axis=0)
            else:
                new_centers[i] = self.cluster_centers_[i]
        
        return new_centers
    
    def _compute_inertia(self, data: np.ndarray) -> float:
        inertia = 0.0
        for i in range(self.n_clusters):
            cluster_points = data[self.labels_ == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.cluster_centers_[i]) ** 2)
        
        return inertia
