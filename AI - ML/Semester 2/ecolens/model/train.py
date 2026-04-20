"""
model/train.py
EcoLens — CNN Model Training using MobileNetV2 Transfer Learning
Trains on TrashNet dataset for 6-class waste classification.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_FROZEN = 10
EPOCHS_FINETUNE = 15
LEARNING_RATE = 1e-3
FINETUNE_LR = 1e-5
NUM_CLASSES = 6
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def create_data_generators():
    """Create augmented data generators for training."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        PROCESSED_DIR / "train",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        PROCESSED_DIR / "val",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=False
    )

    test_gen = val_datagen.flow_from_directory(
        PROCESSED_DIR / "test",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=False
    )

    return train_gen, val_gen, test_gen


def build_model():
    """Build MobileNetV2 transfer learning model."""
    print("\n🏗️  Building MobileNetV2 transfer learning model...")

    # Base model — pretrained on ImageNet
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )
    base_model.trainable = False  # Freeze initially

    # Custom classification head
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc')]
    )

    print(f"  Total params: {model.count_params():,}")
    print(f"  Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    return model, base_model


def train_model(model, base_model, train_gen, val_gen):
    """Two-phase training: frozen base → fine-tuning."""

    # Phase 1: Train classification head only
    print("\n📚 Phase 1: Training classification head (frozen base)...")

    cb_phase1 = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss', verbose=1),
        callbacks.ModelCheckpoint(
            str(MODEL_DIR / "ecolens_phase1.h5"),
            save_best_only=True, monitor='val_accuracy', verbose=1
        )
    ]

    hist1 = model.fit(
        train_gen,
        epochs=EPOCHS_FROZEN,
        validation_data=val_gen,
        callbacks=cb_phase1,
        verbose=1
    )

    print(f"\n✅ Phase 1 complete. Best val accuracy: {max(hist1.history['val_accuracy']):.4f}")

    # Phase 2: Unfreeze top layers for fine-tuning
    print("\n🔧 Phase 2: Fine-tuning top layers of MobileNetV2...")

    # Unfreeze last 30 layers of base model
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(FINETUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc')]
    )

    cb_phase2 = [
        callbacks.EarlyStopping(patience=7, restore_best_weights=True, monitor='val_accuracy'),
        callbacks.ReduceLROnPlateau(factor=0.3, patience=3, monitor='val_loss', verbose=1),
        callbacks.ModelCheckpoint(
            str(MODEL_DIR / "ecolens_best.h5"),
            save_best_only=True, monitor='val_accuracy', verbose=1
        )
    ]

    hist2 = model.fit(
        train_gen,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen,
        callbacks=cb_phase2,
        verbose=1
    )

    print(f"\n✅ Phase 2 complete. Best val accuracy: {max(hist2.history['val_accuracy']):.4f}")
    return hist1, hist2


def evaluate_model(model, test_gen):
    """Evaluate model and compute per-class metrics."""
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n📊 Evaluating on test set...")
    test_gen.reset()

    loss, acc, top2 = model.evaluate(test_gen, verbose=0)
    print(f"  Test Loss: {loss:.4f}")
    print(f"  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Top-2 Accuracy: {top2:.4f}")

    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    return {"loss": loss, "accuracy": acc, "top2_accuracy": top2,
            "report": report, "confusion_matrix": cm.tolist()}


def plot_training_history(hist1, hist2):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('EcoLens Training History', fontsize=16, fontweight='bold')

    # Combine histories
    acc = hist1.history['accuracy'] + hist2.history['accuracy']
    val_acc = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
    loss = hist1.history['loss'] + hist2.history['loss']
    val_loss = hist1.history['val_loss'] + hist2.history['val_loss']
    phase_split = len(hist1.history['accuracy'])

    epochs = range(1, len(acc) + 1)

    # Accuracy
    axes[0].plot(epochs, acc, 'b-', label='Training Accuracy', linewidth=2)
    axes[0].plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    axes[0].axvline(phase_split, color='green', linestyle='--', alpha=0.7, label='Fine-tuning start')
    axes[0].set_title('Model Accuracy', fontsize=13)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)
    axes[1].plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    axes[1].axvline(phase_split, color='green', linestyle='--', alpha=0.7, label='Fine-tuning start')
    axes[1].set_title('Model Loss', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "training_history.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📈 Training history saved to {MODEL_DIR / 'training_history.png'}")


def save_model_metadata(eval_results):
    """Save model metadata for the Streamlit app."""
    metadata = {
        "model_name": "EcoLens Waste Classifier",
        "architecture": "MobileNetV2 + Custom Head",
        "classes": CLASSES,
        "image_size": list(IMAGE_SIZE),
        "trained_at": datetime.now().isoformat(),
        "test_accuracy": eval_results["accuracy"],
        "test_top2_accuracy": eval_results["top2_accuracy"],
        "per_class_metrics": eval_results["report"],
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n💾 Model metadata saved to {MODEL_DIR / 'metadata.json'}")


if __name__ == "__main__":
    print("=" * 60)
    print("  🌿 EcoLens — Model Training Pipeline")
    print("=" * 60)

    # Check dataset
    if not (PROCESSED_DIR / "train").exists():
        print("❌ Dataset not found. Run: python download_dataset.py")
        exit(1)

    # Data
    train_gen, val_gen, test_gen = create_data_generators()
    print(f"\n📂 Dataset Summary:")
    print(f"  Training samples:   {train_gen.n}")
    print(f"  Validation samples: {val_gen.n}")
    print(f"  Test samples:       {test_gen.n}")
    print(f"  Classes: {CLASSES}")

    # Build & train
    model, base_model = build_model()
    hist1, hist2 = train_model(model, base_model, train_gen, val_gen)

    # Evaluate
    eval_results = evaluate_model(model, test_gen)

    # Save artifacts
    model.save(MODEL_DIR / "ecolens_final.h5")
    print(f"\n💾 Final model saved to {MODEL_DIR / 'ecolens_final.h5'}")

    plot_training_history(hist1, hist2)
    save_model_metadata(eval_results)

    print("\n" + "=" * 60)
    print(f"  ✅ Training Complete!")
    print(f"  Final Test Accuracy: {eval_results['accuracy']*100:.2f}%")
    print("=" * 60)
