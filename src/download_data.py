"""
Download and prepare training data for HandScript AI.
Save this file in: HandScriptAI/src/download_data.py
"""

import os
import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image
from tqdm import tqdm

def create_dataset():
    # Create directories
    train_dir = "../data/train"
    test_dir = "../data/test"
    
    for i in range(10):  # 0-9 digit classes
        os.makedirs(f"{train_dir}/{i}", exist_ok=True)
        os.makedirs(f"{test_dir}/{i}", exist_ok=True)
    
    # Load MNIST
    print("Downloading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Save training images
    print("Saving training images...")
    for idx, (img, label) in enumerate(tqdm(zip(x_train, y_train), total=len(x_train))):
        img_pil = Image.fromarray(img)
        img_pil.save(f"{train_dir}/{label}/{idx}.png")
    
    # Save test images
    print("Saving test images...")
    for idx, (img, label) in enumerate(tqdm(zip(x_test, y_test), total=len(x_test))):
        img_pil = Image.fromarray(img)
        img_pil.save(f"{test_dir}/{label}/{idx}.png")
    
    print(f"\nDataset created!")
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_dataset()
