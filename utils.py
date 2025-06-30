import os
import cv2
import numpy as np

# Folder names must exactly match these
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def load_fer2013_from_folder(base_path):
    X, y = [], []
    for idx, emotion in enumerate(emotion_labels):
        emotion_dir = os.path.join(base_path, emotion)
        print(f" Checking: {emotion_dir} -> Exists: {os.path.exists(emotion_dir)}")

        if not os.path.exists(emotion_dir):
            print(f" Folder missing: {emotion_dir}")
            continue

        files = os.listdir(emotion_dir)
        print(f" Found {len(files)} images in '{emotion}'")

        for img_file in files:
            img_path = os.path.join(emotion_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f" Skipped invalid image: {img_path}")
                continue
            img = cv2.resize(img, (48, 48))
            X.append(img)
            y.append(idx)

    X = np.array(X).reshape(-1, 48, 48, 1).astype('float32') / 255.0
    y = np.array(y)
    return X, y
def preprocess_input(face_img):
    import cv2
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.reshape(1, 48, 48, 1)
    face_img = face_img.astype('float32') / 255.0
    return face_img