import os
from collections import Counter
from PIL import Image

DATA_DIR =r"C:\Users\Nitin\Documents\Facial Expression Recognition-project\Dataset"  # adjust path

def get_class_counts(data_dir=DATA_DIR):
    counts = {'train': {}, 'test': {}}
    for split in ['train', 'test']:
        split_dir = os.path.join(data_dir, split)
        if os.path.isdir(split_dir):
            for emotion in os.listdir(split_dir):
                emotion_dir = os.path.join(split_dir, emotion)
                if os.path.isdir(emotion_dir):
                    files = [f for f in os.listdir(emotion_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
                    counts[split][emotion] = len(files)
    return counts

if __name__ == "__main__":
    print(get_class_counts())
