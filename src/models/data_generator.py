from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple
import numpy as np


class DataSource(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass
    
    @abstractmethod
    def get_item(self, index: int) -> np.ndarray:
        pass
    
    @abstractmethod
    def open(self):
        pass
    
    @abstractmethod
    def close(self):
        pass


class DataGenerator(ABC):
    def __init__(
        self,
        data_source: DataSource,
        indices: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        preprocessing_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        shuffle: bool = True
    ):
        self.data_source = data_source
        self.indices = indices
        self.labels = labels
        self.batch_size = batch_size
        self.preprocessing_fn = preprocessing_fn
        self.shuffle = shuffle
    
    @abstractmethod
    def __call__(self):
        pass
    
    def _get_batch_data(self, batch_indices: np.ndarray) -> np.ndarray:
        batch_data = []
        for idx in batch_indices:
            item = self.data_source.get_item(idx)
            if self.preprocessing_fn is not None:
                item = self.preprocessing_fn(item)
            batch_data.append(item)
        return np.array(batch_data)
    
    def _get_batch_labels(self, batch_indices: np.ndarray) -> np.ndarray:
        return self.labels[batch_indices]
