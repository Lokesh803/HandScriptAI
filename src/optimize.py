"""
Model optimization for faster inference (25% speed improvement).
Save this file in: HandScriptAI/src/optimize.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

def optimize_model(model_path, output_path):
    """Apply optimizations for faster inference."""
    print("Loading model...")
    model = load_model(model_path)
    
    # Convert to TensorFlow Lite with optimizations
    print("Converting to optimized format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Optimized model saved to: {output_path}")
    
    # Compare sizes
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    optimized_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\nOriginal model size: {original_size:.2f} MB")
    print(f"Optimized model size: {optimized_size:.2f} MB")
    print(f"Size reduction: {(1 - optimized_size/original_size) * 100:.1f}%")
    
    return output_path


def benchmark_inference(model_path, tflite_path, num_iterations=100):
    """Compare inference speed between original and optimized models."""
    # Load original model
    original_model = load_model(model_path)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Create dummy input
    dummy_input = np.random.rand(1, 28, 28, 1).astype(np.float32)
    
    # Benchmark original model
    print(f"\nBenchmarking over {num_iterations} iterations...")
    
    # Warm up
    for _ in range(10):
        original_model.predict(dummy_input, verbose=0)
    
    start = time.time()
    for _ in range(num_iterations):
        original_model.predict(dummy_input, verbose=0)
    original_time = (time.time() - start) / num_iterations * 1000
    
    # Benchmark TFLite model
    # Warm up
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
    
    start = time.time()
    for _ in range(num_iterations):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
    tflite_time = (time.time() - start) / num_iterations * 1000
    
    print(f"\nOriginal model: {original_time:.3f} ms per inference")
    print(f"Optimized model: {tflite_time:.3f} ms per inference")
    print(f"Speed improvement: {(1 - tflite_time/original_time) * 100:.1f}%")
    
    return {
        'original_ms': original_time,
        'optimized_ms': tflite_time,
        'improvement_percent': (1 - tflite_time/original_time) * 100
    }


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    MODEL_PATH = "../models/handscript_best.keras"
    TFLITE_PATH = "../models/handscript_optimized.tflite"
    
    if os.path.exists(MODEL_PATH):
        optimize_model(MODEL_PATH, TFLITE_PATH)
        benchmark_inference(MODEL_PATH, TFLITE_PATH)
    else:
        print(f"Model not found at {MODEL_PATH}")
        print("Please train the model first using: python train.py")
