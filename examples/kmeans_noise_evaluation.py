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
from src.metrics import ImageMetrics
from src.config import DEFAULT_N_CLUSTERS, RANDOM_SEED, MAX_ITERATIONS, CONVERGENCE_TOLERANCE


def main(image_path: str = 'input_image.jpg', n_clusters: int = DEFAULT_N_CLUSTERS, tolerance: float = CONVERGENCE_TOLERANCE):
    image_path = str(Path(image_path).resolve())
    
    image_processor = ImageProcessor(normalize=True, random_state=RANDOM_SEED)
    visualizer = ResultVisualizer(dpi=150)
    
    print(f"Loading image from '{image_path}'...")
    image = Image.load(image_path)
    
    image_info = image.get_info()
    print(f"Image shape: {image_info['shape']}")
    print(f"Number of pixels: {image_info['total_pixels']:,}")
    
    print("\n" + "=" * 80)
    print("GAUSSIAN NOISE EVALUATION")
    print("=" * 80)
    
    gaussian_intensities = [10.0, 25.0, 50.0]
    
    for std in gaussian_intensities:
        print(f"\n--- Gaussian Noise (std={std}) ---")
        
        noisy_image = image_processor.apply_gaussian_noise(image, mean=0.0, std=std)
        
        pixels = image_processor.prepare_for_clustering(noisy_image)
        
        pbar = tqdm(total=MAX_ITERATIONS, desc=f"Gaussian std={std}", unit="iter", leave=False)
        
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
        
        clustered_image = image_processor.reconstruct_image(
            labels=kmeans.labels_,
            cluster_centers=kmeans.cluster_centers_
        )
        
        print(f"Clustering: {kmeans.n_iter_} iterations, {elapsed_time:.3f}s, inertia={kmeans.inertia_:.2f}")
        
        mse = ImageMetrics.mse(image, clustered_image)
        psnr = ImageMetrics.psnr(image, clustered_image)
        precision = ImageMetrics.precision(image, clustered_image, tolerance=10)
        recall = ImageMetrics.recall(image, clustered_image, tolerance=10)
        f1 = ImageMetrics.f1_score(image, clustered_image, tolerance=10)
        
        print(f"Metrics (vs Original):")
        print(f"  MSE: {mse:.2f}")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  Precision (tol=10): {precision:.4f}")
        print(f"  Recall (tol=10): {recall:.4f}")
        print(f"  F1 Score (tol=10): {f1:.4f}")
        
        title_noisy = f'Gaussian Noise (std={std})'
        title_clustered = (
            f'Clustered (std={std})\n'
            f'{kmeans.n_iter_} iter, {elapsed_time:.3f}s, inertia={kmeans.inertia_:.2f}\n'
            f'MSE={mse:.2f}, PSNR={psnr:.2f}dB\n'
            f'P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}'
        )
        
        visualizer.visualize_comparison(
            original=noisy_image,
            processed=clustered_image,
            title_original=title_noisy,
            title_processed=title_clustered
        )
    
    print("\n" + "=" * 80)
    print("IMPULSE NOISE EVALUATION")
    print("=" * 80)
    
    impulse_intensities = [(0.01, 0.01), (0.03, 0.03), (0.05, 0.05)]
    
    for salt_prob, pepper_prob in impulse_intensities:
        print(f"\n--- Impulse Noise (salt={salt_prob}, pepper={pepper_prob}) ---")
        
        noisy_image = image_processor.apply_impulse_noise(image, salt_prob=salt_prob, pepper_prob=pepper_prob)
        
        pixels = image_processor.prepare_for_clustering(noisy_image)
        
        noise_pct = (salt_prob + pepper_prob) * 100
        pbar = tqdm(total=MAX_ITERATIONS, desc=f"Impulse {noise_pct:.0f}%", unit="iter", leave=False)
        
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
        
        clustered_image = image_processor.reconstruct_image(
            labels=kmeans.labels_,
            cluster_centers=kmeans.cluster_centers_
        )
        
        print(f"Clustering: {kmeans.n_iter_} iterations, {elapsed_time:.3f}s, inertia={kmeans.inertia_:.2f}")
        
        mse = ImageMetrics.mse(image, clustered_image)
        psnr = ImageMetrics.psnr(image, clustered_image)
        precision = ImageMetrics.precision(image, clustered_image, tolerance=10)
        recall = ImageMetrics.recall(image, clustered_image, tolerance=10)
        f1 = ImageMetrics.f1_score(image, clustered_image, tolerance=10)
        
        print(f"Metrics (vs Original):")
        print(f"  MSE: {mse:.2f}")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  Precision (tol=10): {precision:.4f}")
        print(f"  Recall (tol=10): {recall:.4f}")
        print(f"  F1 Score (tol=10): {f1:.4f}")
        
        noise_pct = (salt_prob + pepper_prob) * 100
        title_noisy = f'Impulse Noise ({noise_pct:.0f}%)'
        title_clustered = (
            f'Clustered ({noise_pct:.0f}%)\n'
            f'{kmeans.n_iter_} iter, {elapsed_time:.3f}s, inertia={kmeans.inertia_:.2f}\n'
            f'MSE={mse:.2f}, PSNR={psnr:.2f}dB\n'
            f'P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}'
        )
        
        visualizer.visualize_comparison(
            original=noisy_image,
            processed=clustered_image,
            title_original=title_noisy,
            title_processed=title_clustered
        )
    
    print("\n" + "=" * 80)
    print("✓ All evaluations completed!")
    print("=" * 80)
    visualizer.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='K-Means clustering evaluation with different noise intensities',
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
        '-t', '--tolerance',
        type=float,
        default=CONVERGENCE_TOLERANCE,
        help='Convergence tolerance for K-Means'
    )
    
    args = parser.parse_args()
    main(args.input_image, args.clusters, args.tolerance)
