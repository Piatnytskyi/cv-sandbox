import sys
import argparse
import time
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clustering import KMeans, ForgyInitialization
from src.preprocessing import ImageProcessor
from src.visualization import ResultVisualizer
from src.models import Image
from src.config import DEFAULT_N_CLUSTERS, RANDOM_SEED, MAX_ITERATIONS, CONVERGENCE_TOLERANCE

def main(image_path: str = 'input_image.jpg', n_clusters: int = DEFAULT_N_CLUSTERS, output_path: str = None, tolerance: float = CONVERGENCE_TOLERANCE):
    image_path = str(Path(image_path).resolve())
    
    image_processor = ImageProcessor(normalize=True, random_state=RANDOM_SEED)
    visualizer = ResultVisualizer(dpi=150)
    
    print(f"Loading image from '{image_path}'...")
    image = Image.load(image_path)
    
    image_info = image.get_info()
    print(f"Image shape: {image_info['shape']}")
    print(f"Number of pixels: {image_info['total_pixels']:,}")
    
    print(f"\nPerforming K-Means clustering with {n_clusters} clusters...")
    
    pixels = image_processor.prepare_for_clustering(image)
    
    pbar = tqdm(total=MAX_ITERATIONS, desc="K-Means", unit="iter")
    
    def progress_callback(iteration, center_shift):
        pbar.set_postfix({'shift': f'{center_shift:.2e}'})
        pbar.update(1)
    
    start_time = time.time()
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        initialization_strategy=ForgyInitialization(),
        random_state=RANDOM_SEED,
        tolerance=tolerance,
        progress_callback=progress_callback
    )
    kmeans.fit(pixels)
    
    elapsed_time = time.time() - start_time
    pbar.close()
    
    print(f"\nClustering completed in {kmeans.n_iter_} iterations")
    print(f"Time elapsed: {elapsed_time:.3f} seconds")
    print(f"Inertia: {kmeans.inertia_:.2f}")
    
    quantized_image = image_processor.reconstruct_image(
        labels=kmeans.labels_,
        cluster_centers=kmeans.cluster_centers_
    )
    
    if output_path:
        output_path = str(Path(output_path).resolve())
        quantized_image.save(output_path)
        print(f"\n✓ Clustered image saved to: {output_path}")
    
    title_processed = f'Clustered Image\n({n_clusters} colors, {kmeans.n_iter_} iterations, inertia: {kmeans.inertia_:.2f})'
    
    visualizer.visualize_comparison(
        original=image,
        processed=quantized_image,
        title_original='Original Image',
        title_processed=title_processed
    )
    
    print("\n✓ Visualization created")

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
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output path to save the clustered image'
    )
    parser.add_argument(
        '-t', '--tolerance',
        type=float,
        default=CONVERGENCE_TOLERANCE,
        help='Convergence tolerance for K-Means'
    )
    
    args = parser.parse_args()
    main(args.input_image, args.clusters, args.output, args.tolerance)
