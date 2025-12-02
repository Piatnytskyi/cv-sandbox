import sys
import argparse
from pathlib import Path
import os
import h5py
import kagglehub
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import ImageProcessor
from src.visualization import ResultVisualizer
from src.config import RANDOM_SEED
from src.models import Image

def preprocess_images(images, apply_binarization, apply_erosion, binarization_threshold, erosion_kernel_size, image_processor):
    processed_images = []
    for image in images:
        if apply_binarization:
            img_obj = Image(image)
            img_obj = image_processor.apply_binarization(img_obj, threshold=binarization_threshold)
            image = img_obj.data
        
        if apply_erosion:
            img_obj = Image(image)
            img_obj = image_processor.apply_erosion(img_obj, kernel_size=erosion_kernel_size)
            image = img_obj.data
        
        processed_images.append(image)
    return np.array(processed_images)

def main(
    apply_binarization: bool = False,
    apply_erosion: bool = False,
    binarization_threshold: float = 0.45,
    erosion_kernel_size: int = 3,
    epochs: int = 20
):
    print("=" * 80)
    print("CNN IMAGE CLASSIFICATION - SIGNS DETECTION")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Binarization: {apply_binarization} (threshold={binarization_threshold})")
    print(f"  Erosion: {apply_erosion} (kernel_size={erosion_kernel_size})")
    print(f"  Epochs: {epochs}")
    
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    print("Downloading signs-detection-dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download("maneesh99/signs-detection-dataset")
    print(f"Dataset downloaded to: {dataset_path}")
    
    train_h5_path = Path(dataset_path) / 'Signs_Data_Training.h5'
    test_h5_path = Path(dataset_path) / 'Signs_Data_Testing.h5'
    
    print(f"\nLoading training data from {train_h5_path}")
    train_dataset = h5py.File(train_h5_path, "r")
    X_train_orig = np.array(train_dataset["train_set_x"][:])
    Y_train_orig = np.array(train_dataset["train_set_y"][:])
    
    print(f"Loading test data from {test_h5_path}")
    test_dataset = h5py.File(test_h5_path, "r")
    X_test_orig = np.array(test_dataset["test_set_x"][:])
    Y_test_orig = np.array(test_dataset["test_set_y"][:])
    classes = np.array(test_dataset["list_classes"][:])
    
    train_dataset.close()
    test_dataset.close()
    
    print(f"\nData shapes:")
    print(f"  X_train: {X_train_orig.shape}")
    print(f"  Y_train: {Y_train_orig.shape}")
    print(f"  X_test: {X_test_orig.shape}")
    print(f"  Y_test: {Y_test_orig.shape}")
    print(f"  Number of classes: {len(classes)}")
    print(f"  Classes: {classes}")
    
    print("\n" + "=" * 80)
    print("PREPROCESSING DATA")
    print("=" * 80)
    
    X_train = X_train_orig / 255.0
    X_test = X_test_orig / 255.0
    
    if apply_binarization or apply_erosion:
        print("Applying custom preprocessing...")
        image_processor = ImageProcessor(normalize=False, random_state=RANDOM_SEED)
        X_train = preprocess_images(
            X_train, 
            apply_binarization, 
            apply_erosion, 
            binarization_threshold, 
            erosion_kernel_size, 
            image_processor
        )
        X_test = preprocess_images(
            X_test, 
            apply_binarization, 
            apply_erosion, 
            binarization_threshold, 
            erosion_kernel_size, 
            image_processor
        )
    
    Y_train = tf.keras.utils.to_categorical(Y_train_orig, num_classes=len(classes))
    Y_test = tf.keras.utils.to_categorical(Y_test_orig, num_classes=len(classes))
    
    print(f"\nPreprocessed data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  Y_train: {Y_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  Y_test: {Y_test.shape}")
    
    print("\n" + "=" * 80)
    print("BUILDING MODEL")
    print("=" * 80)
    
    num_classes = len(classes)
    input_shape = (64, 64, 3)
    
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel architecture:")
    model.summary()
    
    print("\n" + "=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)
    
    history = model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        validation_split=0.2,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
    )
    
    print("\n" + "=" * 80)
    print("EVALUATING MODEL")
    print("=" * 80)
    
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)
    
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(Y_test, axis=1)
    
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    
    class_names = [str(cls.decode('utf-8') if isinstance(cls, bytes) else cls) for cls in classes]
    
    print(classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        zero_division=0
    ))
    
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)
    
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    print("\n" + "=" * 80)
    print("DISPLAYING SAMPLE PREDICTIONS")
    print("=" * 80)
    
    n_samples = min(10, len(X_test))
    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
    sample_batch_x = X_test[sample_indices]
    sample_batch_y = Y_test[sample_indices]
    
    sample_pred = model.predict(sample_batch_x, verbose=0)
    sample_pred_indices = np.argmax(sample_pred, axis=1)
    sample_true_indices = np.argmax(sample_batch_y, axis=1)
    
    sample_pred_labels = [class_names[idx] for idx in sample_pred_indices]
    sample_true_labels = [class_names[idx] for idx in sample_true_indices]
    
    visualizer = ResultVisualizer(dpi=150)
    cmap = 'gray' if apply_binarization else None
    visualizer.visualize_predictions(
        images=sample_batch_x,
        pred_labels=sample_pred_labels,
        true_labels=sample_true_labels,
        title='Sample Predictions on Signs Detection Dataset',
        figsize=(15, 6),
        cmap=cmap
    )
    visualizer.show()
    
    print("\n" + "=" * 80)
    print("✓ Classification completed!")
    print("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CNN image classification on Signs Detection dataset',
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
        default=0.45,
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
        default=20,
        help='Number of training epochs'
    )
    args = parser.parse_args()
    
    main(
        apply_binarization=args.binarize,
        apply_erosion=args.erode,
        binarization_threshold=args.binarization_threshold,
        erosion_kernel_size=args.erosion_kernel_size,
        epochs=args.epochs
    )
