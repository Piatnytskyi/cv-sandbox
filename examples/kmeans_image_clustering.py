import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clustering import KMeans, ForgyInitialization
from src.preprocessing import ImageProcessor
from src.visualization import ResultVisualizer
from src.models import Image
from src.config import DEFAULT_N_CLUSTERS, RANDOM_SEED


def main(image_path: str = 'input_image.jpg', n_clusters: int = DEFAULT_N_CLUSTERS):
    image_path = str(Path(image_path).resolve())
    
    image_processor = ImageProcessor(normalize=True)
    visualizer = ResultVisualizer(dpi=150)
    
    print(f"Loading image from '{image_path}'...")
    image = Image.load(image_path)
    
    pixels = image_processor.prepare_for_clustering(image)
    
    image_info = image.get_info()
    print(f"Image shape: {image_info['shape']}")
    print(f"Number of pixels: {image_info['total_pixels']:,}")
    
    print(f"\nPerforming K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        initialization_strategy=ForgyInitialization(),
        random_state=RANDOM_SEED
    )
    
    kmeans.fit(pixels)
    
    print(f"Clustering completed in {kmeans.n_iter_} iterations")
    print(f"Inertia: {kmeans.inertia_:.2f}")
    
    quantized_image = image_processor.reconstruct_image(
        labels=kmeans.labels_,
        cluster_centers=kmeans.cluster_centers_
    )

    title_parts = [f'{n_clusters} colors', f'{kmeans.n_iter_} iterations', f'inertia: {kmeans.inertia_:.2f}']
    title_processed = f'Clustered Image\n({", ".join(title_parts)})'

    visualizer.visualize_comparison(
        original=image_processor.original_image,
        processed=quantized_image,
        title_original='Original Image',
        title_processed=title_processed
    )
    
    visualizer.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='K-Means clustering for image color quantization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'input_image',
        nargs='?',
        default='input_image.jpg',
        help='Path to the input image file'
    )
    parser.add_argument(
        '-k', '--clusters',
        type=int,
        default=DEFAULT_N_CLUSTERS,
        help='Number of clusters for K-Means'
    )
    
    args = parser.parse_args()
    main(args.input_image, args.clusters)
