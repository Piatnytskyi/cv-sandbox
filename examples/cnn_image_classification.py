import sys
import argparse
from pathlib import Path
import h5py
import kagglehub
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import ImageProcessor
from src.visualization import ResultVisualizer
from src.config import RANDOM_SEED

def main(
    apply_binarization: bool = False,
    apply_erosion: bool = False,
    binarization_threshold: float = 0.3,
    erosion_kernel_size: int = 3,
    epochs: int = 5,
    batch_size: int = 128
):
    print("=" * 80)
    print("CNN IMAGE CLASSIFICATION - FOOD-101")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Binarization: {apply_binarization} (threshold={binarization_threshold})")
    print(f"  Erosion: {apply_erosion} (kernel_size={erosion_kernel_size})")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    print("Downloading Food-41 dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download("kmader/food41")
    print(f"Dataset downloaded to: {dataset_path}")
    
    h5_file_path = Path(dataset_path) / "food_c101_n1000_r384x384x3.h5"
    print(f"Loading data from: {h5_file_path}")
    
    with h5py.File(h5_file_path, 'r') as h5file:
        images = h5file['images'][:]
        category_onehot = h5file['category'][:]
        category_names_bytes = h5file['category_names'][:]
        
        print(f"\nDataset loaded successfully:")
        print(f"  Images shape: {images.shape}")
        print(f"  Category (one-hot) shape: {category_onehot.shape}")
        print(f"  Number of categories: {len(category_names_bytes)}")

    labels = category_onehot.argmax(axis=1)
    class_names = [name.decode('utf-8') for name in category_names_bytes]
    
    print(f"\nClass names sample (first 10): {class_names[:10]}")
    print(f"Labels range: {labels.min()} to {labels.max()}")
    print(f"Unique labels: {len(np.unique(labels))}")

    print("\n" + "=" * 80)
    print("SPLITTING DATA")
    print("=" * 80)
    
    train_indices, test_indices = train_test_split(
        np.arange(len(images)), 
        test_size=0.2, 
        random_state=RANDOM_SEED, 
        stratify=labels
    )
    
    train_images = images[train_indices]
    test_images = images[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    
    print(f"Training set: {train_images.shape[0]} images")
    print(f"Test set: {test_images.shape[0]} images")
    print(f"Image shape: {train_images.shape[1:]}")
    
    if apply_binarization or apply_erosion:
        print("\n" + "=" * 80)
        print("PREPROCESSING IMAGES")
        print("=" * 80)
        
        image_processor = ImageProcessor(normalize=False, random_state=RANDOM_SEED)
        
        if apply_binarization:
            print(f"Applying binarization (threshold={binarization_threshold})...")
            
            from src.models import Image
            train_images_processed = []
            for img in train_images:
                img_uint8 = (img * 255).astype(np.uint8)
                img_obj = Image(img_uint8)
                binary_img = image_processor.apply_binarization(img_obj, threshold=binarization_threshold)
                train_images_processed.append(binary_img.data.astype(np.float32) / 255.0)
            train_images = np.array(train_images_processed)
            
            test_images_processed = []
            for img in test_images:
                img_uint8 = (img * 255).astype(np.uint8)
                img_obj = Image(img_uint8)
                binary_img = image_processor.apply_binarization(img_obj, threshold=binarization_threshold)
                test_images_processed.append(binary_img.data.astype(np.float32) / 255.0)
            test_images = np.array(test_images_processed)
            
            print("  Binarization completed")
        
        if apply_erosion:
            print(f"Applying erosion (kernel_size={erosion_kernel_size})...")
            
            from src.models import Image
            train_images_processed = []
            for img in train_images:
                img_uint8 = (img * 255).astype(np.uint8)
                img_obj = Image(img_uint8)
                eroded_img = image_processor.apply_erosion(img_obj, kernel_size=erosion_kernel_size)
                train_images_processed.append(eroded_img.data.astype(np.float32) / 255.0)
            train_images = np.array(train_images_processed)
            
            test_images_processed = []
            for img in test_images:
                img_uint8 = (img * 255).astype(np.uint8)
                img_obj = Image(img_uint8)
                eroded_img = image_processor.apply_erosion(img_obj, kernel_size=erosion_kernel_size)
                test_images_processed.append(eroded_img.data.astype(np.float32) / 255.0)
            test_images = np.array(test_images_processed)
            
            print("  Erosion completed")
    
    print("\n" + "=" * 80)
    print("BUILDING MODEL")
    print("=" * 80)
    
    num_classes = len(class_names)
    input_shape = train_images.shape[1:]
    
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel architecture:")
    model.summary()
    
    print("\n" + "=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)
    
    history = model.fit(
        train_images,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )
    
    print("\n" + "=" * 80)
    print("EVALUATING MODEL")
    print("=" * 80)
    
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    y_pred = model.predict(test_images, verbose=0).argmax(axis=1)
    
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT (Top 20 classes)")
    print("=" * 80)
    
    unique_labels, counts = np.unique(test_labels, return_counts=True)
    top_20_indices = unique_labels[np.argsort(counts)[-20:]]
    
    mask = np.isin(test_labels, top_20_indices)
    filtered_test_labels = test_labels[mask]
    filtered_pred_labels = y_pred[mask]
    
    top_20_class_names = [class_names[i] for i in top_20_indices]
    
    print(classification_report(
        filtered_test_labels, 
        filtered_pred_labels, 
        labels=top_20_indices,
        target_names=top_20_class_names,
        zero_division=0
    ))
    
    print("\n" + "=" * 80)
    print("DISPLAYING SAMPLE PREDICTIONS")
    print("=" * 80)
    
    n_samples = 10
    indices = np.random.RandomState(RANDOM_SEED).choice(test_images.shape[0], n_samples, replace=False)
    
    sample_images = test_images[indices]
    sample_pred_labels = [class_names[y_pred[idx]] for idx in indices]
    sample_true_labels = [class_names[test_labels[idx]] for idx in indices]
    
    visualizer = ResultVisualizer(dpi=150)
    visualizer.visualize_predictions(
        images=sample_images,
        pred_labels=sample_pred_labels,
        true_labels=sample_true_labels,
        title='Sample Predictions on Food-101 Dataset',
        figsize=(15, 6),
        cmap=None
    )
    visualizer.show()
    
    print("\n" + "=" * 80)
    print("✓ Classification completed!")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CNN image classification on Food-101 dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--binarize',
        action='store_true',
        help='Apply binarization to images'
    )
    parser.add_argument(
        '--erode',
        action='store_true',
        help='Apply morphological erosion to images'
    )
    parser.add_argument(
        '--binarization-threshold',
        type=float,
        default=0.3,
        help='Threshold for binarization (0.0 to 1.0)'
    )
    parser.add_argument(
        '--erosion-kernel-size',
        type=int,
        default=3,
        help='Kernel size for erosion (odd number)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Training batch size'
    )
    
    args = parser.parse_args()
    
    main(
        apply_binarization=args.binarize,
        apply_erosion=args.erode,
        binarization_threshold=args.binarization_threshold,
        erosion_kernel_size=args.erosion_kernel_size,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
