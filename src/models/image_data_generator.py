from typing import Callable, Optional
import numpy as np
import h5py
from .data_generator import DataSource, DataGenerator


class HDF5ImageSource(DataSource):
    def __init__(self, file_path: str, dataset_name: str = 'images'):
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.file_handle = None
        self.dataset = None
    
    def __len__(self) -> int:
        if self.dataset is not None:
            return len(self.dataset)
        with h5py.File(self.file_path, 'r') as f:
            return len(f[self.dataset_name])
    
    def get_item(self, index: int) -> np.ndarray:
        if self.dataset is None:
            raise RuntimeError("Data source not opened. Call open() first.")
        return self.dataset[index]
    
    def open(self):
        self.file_handle = h5py.File(self.file_path, 'r')
        self.dataset = self.file_handle[self.dataset_name]
    
    def close(self):
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None
            self.dataset = None


class ImageDataGenerator(DataGenerator):
    def __call__(self):
        self.data_source.open()
        
        shuffled_indices = self.indices.copy()
        if self.shuffle:
            np.random.shuffle(shuffled_indices)
        
        for i in range(0, len(shuffled_indices), self.batch_size):
            batch_indices = shuffled_indices[i:i + self.batch_size]
            
            batch_images = self._get_batch_data(batch_indices)
            batch_labels = self._get_batch_labels(batch_indices)
            
            yield batch_images, batch_labels
        
        self.data_source.close()
