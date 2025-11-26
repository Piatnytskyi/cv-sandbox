import sys
import argparse
from pathlib import Path
import os
import pandas as pd
import kagglehub
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import ImageProcessor
from src.visualization import ResultVisualizer
from src.config import RANDOM_SEED
from src.models import Image

def preprocess_function(image, apply_binarization, apply_erosion, binarization_threshold, erosion_kernel_size, image_processor):
    img_uint8 = (image * 255).astype(np.uint8)
    
    if apply_binarization:
        img_obj = Image(img_uint8)
        img_obj = image_processor.apply_binarization(img_obj, threshold=binarization_threshold)
        img_uint8 = img_obj.data
    
    if apply_erosion:
        img_obj = Image(img_uint8)
        img_obj = image_processor.apply_erosion(img_obj, kernel_size=erosion_kernel_size)
        img_uint8 = img_obj.data
    
    return img_uint8.astype(np.float32) / 255.0

def main(
    apply_binarization: bool = False,
    apply_erosion: bool = False,
    binarization_threshold: float = 0.3,
    erosion_kernel_size: int = 3,
    epochs: int = 20,
    batch_size: int = 32,
    samples_per_category: int = 100
):
    print("=" * 80)
    print("CNN IMAGE CLASSIFICATION - FOOD-41")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Samples per category: {samples_per_category}")
    print(f"  Batch size: {batch_size}")
    print(f"  Binarization: {apply_binarization} (threshold={binarization_threshold})")
    print(f"  Erosion: {apply_erosion} (kernel_size={erosion_kernel_size})")
    print(f"  Epochs: {epochs}")
    
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    print("Downloading Food-41 dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download("kmader/food41")
    print(f"Dataset downloaded to: {dataset_path}")
    
    image_dir = Path(dataset_path) / 'images'
    print(f"Image directory: {image_dir}")
    
    filepaths = list(image_dir.glob(r'**/*.jpg'))
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
    
    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')
    
    images_df = pd.concat([filepaths, labels], axis=1)
    
    print(f"\nTotal images found: {len(images_df)}")
    print(f"Number of categories: {images_df['Label'].nunique()}")
    
    print("\n" + "=" * 80)
    print("SAMPLING DATA")
    print("=" * 80)
    
    category_samples = []
    for category in images_df['Label'].unique():
        category_slice = images_df.query("Label == @category")
        sample_size = min(samples_per_category, len(category_slice))
        category_samples.append(category_slice.sample(sample_size, random_state=RANDOM_SEED))
    image_df = pd.concat(category_samples, axis=0).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print(f"Sampled {len(image_df)} images")
    print(f"Label distribution:")
    print(image_df['Label'].value_counts().head(10))
    
    print("\n" + "=" * 80)
    print("SPLITTING DATA")
    print("=" * 80)
    
    train_df, test_df = train_test_split(
        image_df, 
        train_size=0.7, 
        shuffle=True, 
        random_state=RANDOM_SEED
    )
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    print("\n" + "=" * 80)
    print("CREATING DATA GENERATORS")
    print("=" * 80)
    
    target_size = (224, 224)
    
    preprocessing_fn = None
    if apply_binarization or apply_erosion:
        print("Preprocessing will be applied to images")
        image_processor = ImageProcessor(normalize=False, random_state=RANDOM_SEED)
        
        def preprocessing_fn(x):
            result = preprocess_function(
                x, 
                apply_binarization, 
                apply_erosion, 
                binarization_threshold, 
                erosion_kernel_size, 
                image_processor
            )
            return result * 255
    
    train_generator = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        preprocessing_function=preprocessing_fn
    )
    
    test_generator = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=preprocessing_fn
    )
    
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=target_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        seed=RANDOM_SEED,
        subset='training'
    )
    
    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=target_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        seed=RANDOM_SEED,
        subset='validation'
    )
    
    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=target_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"\nGenerator created:")
    print(f"  Training batches: {len(train_images)}")
    print(f"  Validation batches: {len(val_images)}")
    print(f"  Test batches: {len(test_images)}")
    
    print("\n" + "=" * 80)
    print("BUILDING MODEL")
    print("=" * 80)
    
    num_classes = 101
    input_shape = (*target_size, 3)
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
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
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel architecture:")
    model.summary()
    
    print("\n" + "=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)
    
    history = model.fit(
        train_images,
        validation_data=val_images,
        epochs=epochs,
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
    
    test_loss, test_acc = model.evaluate(test_images, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)
    
    y_pred = np.argmax(model.predict(test_images, verbose=0), axis=1)
    y_true = test_images.classes
    
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    
    class_names = list(test_images.class_indices.keys())
    
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
    
    n_samples = min(10, len(test_df))
    test_images.reset()
    sample_batch_x, sample_batch_y = next(test_images)
    sample_batch_x = sample_batch_x[:n_samples]
    sample_batch_y = sample_batch_y[:n_samples]
    
    sample_pred = model.predict(sample_batch_x, verbose=0)
    sample_pred_indices = np.argmax(sample_pred, axis=1)
    sample_true_indices = np.argmax(sample_batch_y, axis=1)
    
    sample_pred_labels = [class_names[idx] for idx in sample_pred_indices]
    sample_true_labels = [class_names[idx] for idx in sample_true_indices]
    
    visualizer = ResultVisualizer(dpi=150)
    visualizer.visualize_predictions(
        images=sample_batch_x,
        pred_labels=sample_pred_labels,
        true_labels=sample_true_labels,
        title='Sample Predictions on Food-41 Dataset',
        figsize=(15, 6),
        cmap=None
    )
    visualizer.show()
    
    print("\n" + "=" * 80)
    print("✓ Classification completed!")
    print("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CNN image classification on Food-41 dataset',
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
        default=20,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--samples-per-category',
        type=int,
        default=100,
        help='Number of samples per category to use'
    )
    args = parser.parse_args()
    
    main(
        apply_binarization=args.binarize,
        apply_erosion=args.erode,
        binarization_threshold=args.binarization_threshold,
        erosion_kernel_size=args.erosion_kernel_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_category=args.samples_per_category
    )
