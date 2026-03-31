import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class ImprovedFacialExpressionRecognizer:
    def __init__(self, model_path, img_size=(48, 48)):
        self.model = load_model(model_path)
        self.img_size = img_size
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Try multiple face detectors for better detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        print("Model loaded successfully!")
        print("Emotion categories:", self.emotions)
        
        # Test model with sample input
        self._test_model()
    
    def _test_model(self):
        """Test if model produces varied outputs"""
        test_input = np.random.normal(0.5, 0.1, (1, 48, 48, 1)).astype('float32')
        predictions = self.model.predict(test_input, verbose=0)
        print("Model test - predictions:", predictions[0])
    
    def enhance_face_contrast(self, face_roi):
        """Enhance contrast to make features more visible"""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(face_roi)
        return enhanced
    
    def preprocess_face(self, face_roi):
        """
        Improved preprocessing with contrast enhancement and normalization
        """
        # Convert to grayscale if needed
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi
        
        # Enhance contrast
        enhanced = self.enhance_face_contrast(gray)
        
        # Resize to model input size
        resized = cv2.resize(enhanced, self.img_size)
        
        # Normalize to 0-1 range
        normalized = resized.astype('float32') / 255.0
        
        # Apply additional preprocessing (same as training)
        # Subtract mean and divide by std (if your training did this)
        normalized = (normalized - 0.5) / 0.5
        
        # Reshape for model
        processed = normalized.reshape(1, self.img_size[0], self.img_size[1], 1)
        
        return processed
    
    def predict_emotion(self, face_roi):
        """Predict emotion with confidence threshold"""
        try:
            processed_face = self.preprocess_face(face_roi)
            predictions = self.model.predict(processed_face, verbose=0)
            
            emotion_idx = np.argmax(predictions[0])
            confidence = predictions[0][emotion_idx]
            emotion = self.emotions[emotion_idx]
            
            # Apply confidence threshold - if too low, show "Analyzing..."
            if confidence < 0.3:  # Adjust this threshold as needed
                emotion = "Analyzing..."
                confidence = 0.0
            
            return emotion, confidence, predictions[0]
            
        except Exception as e:
            return "Error", 0.0, np.zeros(len(self.emotions))
    
    def detect_faces_improved(self, frame):
        """Improved face detection with multiple methods"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with different parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,  # Increased for better accuracy
            minSize=(50, 50),  # Increased minimum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Filter faces by size and aspect ratio
        filtered_faces = []
        for (x, y, w, h) in faces:
            aspect_ratio = w / h
            if 0.5 < aspect_ratio < 2.0:  # Reasonable face aspect ratio
                filtered_faces.append((x, y, w, h))
        
        return filtered_faces
    
    def draw_emotion_info(self, frame, face_bbox, emotion, confidence, all_probabilities=None):
        """Improved visualization"""
        x, y, w, h = face_bbox
        
        # Color based on emotion
        color_map = {
            'Angry': (0, 0, 255),      # Red
            'Happy': (0, 255, 0),      # Green
            'Sad': (255, 0, 0),        # Blue
            'Surprise': (0, 255, 255), # Yellow
            'Fear': (0, 165, 255),     # Orange
            'Disgust': (128, 0, 128),  # Purple
            'Neutral': (128, 128, 128),# Gray
            'Analyzing...': (255, 255, 255), # White
            'Error': (0, 0, 0)         # Black
        }
        
        color = color_map.get(emotion, (255, 255, 255))
        
        # Draw face bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw emotion label
        if emotion in ['Analyzing...', 'Error']:
            label = emotion
        else:
            label = f"{emotion}: {confidence:.2f}"
        
        # Background for text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x, y - text_size[1] - 10), (x + text_size[0], y), color, -1)
        cv2.putText(frame, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw probability bars if provided
        if all_probabilities is not None and emotion not in ['Analyzing...', 'Error']:
            self.draw_probability_bars(frame, x, y + h + 10, all_probabilities)
    
    def draw_probability_bars(self, frame, x, y, probabilities, bar_width=100, bar_height=15):
        """Draw probability bars for all emotions"""
        colors = [(0, 0, 255), (128, 0, 128), (0, 165, 255), 
                 (0, 255, 0), (255, 0, 0), (0, 255, 255), (128, 128, 128)]
        
        for i, (emotion, prob, color) in enumerate(zip(self.emotions, probabilities, colors)):
            # Draw bar background
            cv2.rectangle(frame, (x, y + i*25), (x + bar_width, y + i*25 + bar_height), (50, 50, 50), -1)
            
            # Draw probability bar
            bar_length = int(prob * bar_width)
            cv2.rectangle(frame, (x, y + i*25), (x + bar_length, y + i*25 + bar_height), color, -1)
            
            # Draw text
            text = f"{emotion}: {prob:.2f}"
            cv2.putText(frame, text, (x + bar_width + 5, y + i*25 + bar_height - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run_webcam_demo(self):
        """Run improved real-time detection"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting webcam...")
        print("Press 'q' to quit, 'r' to reset statistics")
        
        # Statistics
        emotion_count = {emotion: 0 for emotion in self.emotions}
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # Detect faces
            faces = self.detect_faces_improved(frame)
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                
                # Predict emotion
                emotion, confidence, all_probs = self.predict_emotion(face_roi)
                
                # Update statistics
                if emotion in emotion_count:
                    emotion_count[emotion] += 1
                
                # Draw results
                self.draw_emotion_info(frame, (x, y, w, h), emotion, confidence, all_probs)
            
            # Display statistics
            self.draw_statistics(frame, emotion_count, frame_count)
            
            # Display frame
            cv2.imshow('Facial Expression Recognition - Improved', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                emotion_count = {emotion: 0 for emotion in self.emotions}
                frame_count = 0
                print("Statistics reset")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n=== Final Statistics ===")
        total_detections = sum(emotion_count.values())
        if total_detections > 0:
            for emotion, count in emotion_count.items():
                percentage = (count / total_detections) * 100
                print(f"{emotion}: {count} ({percentage:.1f}%)")
    
    def draw_statistics(self, frame, emotion_count, frame_count):
        """Draw statistics on frame"""
        y_offset = 30
        cv2.putText(frame, "Statistics:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        total_detections = sum(emotion_count.values())
        for emotion, count in emotion_count.items():
            y_offset += 20
            if total_detections > 0:
                percentage = (count / total_detections) * 100
                text = f"{emotion}: {percentage:.1f}%"
            else:
                text = f"{emotion}: 0%"
            
            color = {
                'Angry': (0, 0, 255), 'Happy': (0, 255, 0), 'Sad': (255, 0, 0),
                'Surprise': (0, 255, 255), 'Fear': (0, 165, 255), 
                'Disgust': (128, 0, 128), 'Neutral': (128, 128, 128)
            }.get(emotion, (255, 255, 255))
            
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def main():
    MODEL_PATH = "best_model.h5"  # Update with your model path
    
    try:
        recognizer = ImprovedFacialExpressionRecognizer(MODEL_PATH)
        recognizer.run_webcam_demo()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()