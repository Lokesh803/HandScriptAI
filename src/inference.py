"""
Real-time inference and document scanning for HandScript AI.
Save this file in: HandScriptAI/src/inference.py
"""

import os
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from preprocessing import ImagePreprocessor

class HandScriptDetector:
    def __init__(self, model_path):
        """Initialize detector with trained model."""
        print(f"Loading model from: {model_path}")
        self.model = load_model(model_path)
        self.preprocessor = ImagePreprocessor(target_size=(28, 28))
        self.class_labels = [str(i) for i in range(10)]
        print("Model loaded successfully!")
    
    def detect_characters(self, image):
        """Detect individual characters in an image."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply preprocessing
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Extract bounding boxes
        detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small noise
            if w > 5 and h > 5:
                detections.append({
                    'bbox': (x, y, w, h),
                    'contour': contour
                })
        
        # Sort by x-coordinate (left to right)
        detections.sort(key=lambda d: d['bbox'][0])
        
        return detections, binary
    
    def predict_character(self, image):
        """Predict a single character from image."""
        # Preprocess
        processed = self.preprocessor.preprocess(image)
        
        # Add batch and channel dimensions
        input_tensor = processed.reshape(1, 28, 28, 1)
        
        # Predict
        start_time = time.time()
        predictions = self.model.predict(input_tensor, verbose=0)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Get result
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return {
            'class': self.class_labels[predicted_class],
            'confidence': float(confidence),
            'inference_time_ms': inference_time
        }
    
    def process_document(self, image_path):
        """Process entire document and extract text."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original = image.copy()
        
        # Detect characters
        detections, binary = self.detect_characters(image)
        
        results = []
        total_inference_time = 0
        
        for det in detections:
            x, y, w, h = det['bbox']
            
            # Extract character region
            char_roi = binary[y:y+h, x:x+w]
            
            # Add padding
            padding = 4
            char_padded = cv2.copyMakeBorder(
                char_roi, padding, padding, padding, padding,
                cv2.BORDER_CONSTANT, value=0
            )
            
            # Predict
            prediction = self.predict_character(char_padded)
            total_inference_time += prediction['inference_time_ms']
            
            # Draw on image
            color = (0, 255, 0) if prediction['confidence'] > 0.8 else (0, 165, 255)
            cv2.rectangle(original, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                original,
                f"{prediction['class']} ({prediction['confidence']:.0%})",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
            
            results.append({
                'bbox': (x, y, w, h),
                **prediction
            })
        
        return {
            'results': results,
            'annotated_image': original,
            'total_inference_time_ms': total_inference_time,
            'num_characters': len(results),
            'extracted_text': ''.join([r['class'] for r in results])
        }
    
    def run_webcam(self):
        """Run real-time detection from webcam."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Press 'q' to quit, 'c' to capture and analyze")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display instructions
            cv2.putText(
                frame, "Press 'c' to capture, 'q' to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            cv2.imshow('HandScript AI - Live', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Process current frame
                detections, binary = self.detect_characters(frame)
                
                for det in detections:
                    x, y, w, h = det['bbox']
                    char_roi = binary[y:y+h, x:x+w]
                    
                    padding = 4
                    char_padded = cv2.copyMakeBorder(
                        char_roi, padding, padding, padding, padding,
                        cv2.BORDER_CONSTANT, value=0
                    )
                    
                    prediction = self.predict_character(char_padded)
                    
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(
                        frame, prediction['class'],
                        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                    )
                
                cv2.imshow('HandScript AI - Detection', frame)
                cv2.waitKey(0)
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main inference function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='HandScript AI Inference')
    parser.add_argument('--model', type=str, default='../models/handscript_best.keras',
                       help='Path to trained model')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image for processing')
    parser.add_argument('--webcam', action='store_true',
                       help='Run webcam detection')
    parser.add_argument('--output', type=str, default='../output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = HandScriptDetector(args.model)
    
    if args.webcam:
        detector.run_webcam()
    elif args.image:
        # Process single image
        result = detector.process_document(args.image)
        
        print(f"\nResults:")
        print(f"Characters detected: {result['num_characters']}")
        print(f"Extracted text: {result['extracted_text']}")
        print(f"Total inference time: {result['total_inference_time_ms']:.2f} ms")
        print(f"Average per character: {result['total_inference_time_ms']/max(1, result['num_characters']):.2f} ms")
        
        # Save annotated image
        output_path = os.path.join(args.output, 'detection_result.png')
        cv2.imwrite(output_path, result['annotated_image'])
        print(f"\nAnnotated image saved to: {output_path}")
    else:
        print("Please specify --image or --webcam")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
