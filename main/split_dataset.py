import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def diagnose_model(model_path):
    """
    Diagnose why the model always predicts the same emotion
    """
    model = load_model(model_path)
    
    # Test with different types of inputs
    test_cases = [
        ('Random noise', np.random.random((1, 48, 48, 1))),
        ('All zeros', np.zeros((1, 48, 48, 1))),
        ('All ones', np.ones((1, 48, 48, 1))),
        ('Medium gray', np.full((1, 48, 48, 1), 0.5)),
    ]
    
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    print("=== MODEL DIAGNOSIS ===")
    for name, test_input in test_cases:
        predictions = model.predict(test_input, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        
        print(f"\n{name}:")
        print(f"Predicted: {emotions[predicted_idx]} ({confidence:.4f})")
        print("All probabilities:")
        for i, (emotion, prob) in enumerate(zip(emotions, predictions[0])):
            print(f"  {emotion}: {prob:.4f}")

# Run diagnosis
diagnose_model(r"C:\Users\Nitin\Documents\Facial Expression Recognition-project\Dataset\pretrained\face_model.h5")