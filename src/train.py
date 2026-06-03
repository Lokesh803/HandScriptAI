"""
Training script for HandScript AI.
Save this file in: HandScriptAI/src/train.py
Run from: HandScriptAI/src/ directory
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import create_handscript_cnn, compile_model, get_callbacks, print_model_summary
from preprocessing import load_and_preprocess_data

def plot_training_history(history, save_path):
    """Plot and save training metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to: {save_path}")


def train():
    """Main training function."""
    print("=" * 60)
    print("HandScript AI - Training Pipeline")
    print("=" * 60)
    
    # Configuration
    TRAIN_DIR = "../data/train"
    TEST_DIR = "../data/test"
    MODEL_DIR = "../models"
    OUTPUT_DIR = "../output"
    
    INPUT_SHAPE = (28, 28, 1)
    NUM_CLASSES = 10
    BATCH_SIZE = 64
    EPOCHS = 50
    
    # Create directories if needed
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading and preprocessing data...")
    train_gen, test_gen = load_and_preprocess_data(
        TRAIN_DIR, TEST_DIR,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE
    )
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    print(f"Number of classes: {len(train_gen.class_indices)}")
    
    # Create model
    print("\n[2/5] Building CNN model...")
    model = create_handscript_cnn(INPUT_SHAPE, NUM_CLASSES)
    model = compile_model(model)
    print_model_summary(model)
    
    # Get callbacks
    print("\n[3/5] Setting up training callbacks...")
    callbacks = get_callbacks(os.path.join(MODEL_DIR, "handscript_best.keras"))
    
    # Train model
    print("\n[4/5] Starting training...")
    print("-" * 40)
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=test_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n[5/5] Evaluating model...")
    print("-" * 40)
    
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(MODEL_DIR, f"handscript_final_{timestamp}.keras")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Plot training history
    plot_path = os.path.join(OUTPUT_DIR, f"training_history_{timestamp}.png")
    plot_training_history(history, plot_path)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {max(history.history['val_accuracy']) * 100:.2f}%")
    print("=" * 60)
    
    return model, history


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    train()

