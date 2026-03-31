import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque

class FacialExpressionPredictor:
    def __init__(self, model_path=r"C:\Users\Nitin\Documents\Facial Expression Recognition-project\Dataset\pretrained\face_model.h5",
                 confidence_threshold=0.5):
        """Initialize the predictor with pretrained model"""
        try:
            self.model = load_model(model_path)
            # Warm up the model
            self.model.predict(np.zeros((1, 48, 48, 1)))
            print("Pretrained model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Resize to model input size (48x48)
        resized = cv2.resize(gray, (48, 48))
        
        # Normalize pixel values
        normalized = resized.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        processed = normalized.reshape(1, 48, 48, 1)
        return processed
    
    def predict_emotion(self, image):
        """Predict emotion from image"""
        processed = self.preprocess_image(image)
        prediction = self.model.predict(processed, verbose=0)[0]
        emotion_idx = np.argmax(prediction)
        return {
            'emotion': self.emotions[emotion_idx],
            'confidence': float(prediction[emotion_idx]),
            'all_probabilities': dict(zip(self.emotions, prediction.tolist()))
        }
    
    def detect_and_predict(self, frame):
        """Detect faces and predict emotions in frame"""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        results = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            prediction = self.predict_emotion(face_roi)
            results.append({
                'bbox': (x, y, w, h),
                **prediction
            })
        return results

    def run_webcam(self):
        """Run real-time facial expression recognition"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        print("Starting webcam detection... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
                
            # Mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect faces and predict emotions
            results = self.detect_and_predict(frame)
            
            # Draw results
            for result in results:
                x, y, w, h = result['bbox']
                emotion = result['emotion']
                conf = result['confidence']
                
                # Draw face box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw emotion label
                label = f"{emotion}: {conf:.2f}"
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Facial Expression Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

def test_image(image_path, model_path=None):
    """Test the model on a single image"""
    predictor = FacialExpressionPredictor(model_path)
    
    # Read and process image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
        
    results = predictor.detect_and_predict(image)
    
    # Draw results on image
    for result in results:
        x, y, w, h = result['bbox']
        emotion = result['emotion']
        conf = result['confidence']
        
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{emotion}: {conf:.2f}"
        cv2.putText(image, label, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Show results
    cv2.imshow('Prediction Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize predictor with pretrained model
    predictor = FacialExpressionPredictor()
    
    # Run webcam demo
    predictor.run_webcam()
    
    # Alternatively, test on a single image:
    # test_image('path/to/your/image.jpg')