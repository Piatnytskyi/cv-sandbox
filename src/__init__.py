__version__ = "1.0.0"

from .clustering import KMeans, AbstractClustering, InitializationStrategy, ForgyInitialization
from .preprocessing import ImageProcessor
from .visualization import ResultVisualizer
from .models import Image
from .config import (
    MAX_ITERATIONS,
    CONVERGENCE_TOLERANCE,
    DEFAULT_N_CLUSTERS,
    RANDOM_SEED,
)

__all__ = [
    "KMeans",
    "AbstractClustering",
    "InitializationStrategy",
    "ForgyInitialization",
    "ImageProcessor",
    "ResultVisualizer",
    "Image",
    "MAX_ITERATIONS",
    "CONVERGENCE_TOLERANCE",
    "DEFAULT_N_CLUSTERS",
    "RANDOM_SEED",
]
