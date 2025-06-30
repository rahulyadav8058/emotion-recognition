import cv2
import numpy as np
import time
import pyttsx3
from tensorflow.keras.models import load_model
from utils import preprocess_input

# Load model and labels
model = load_model('model/emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Voice engine
engine = pyttsx3.init()
last_spoken_emotion = ""
last_spoken_time = 0

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam and timer
cap = cv2.VideoCapture(0)
start_time = time.time()
timeout_seconds = 60

while True:
    elapsed_time = time.time() - start_time
    remaining_time = max(0, int(timeout_seconds - elapsed_time))

    if elapsed_time > timeout_seconds:
        print("Timeout: 1 minute passed. Exiting.")
        break

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = preprocess_input(face)
        prediction = model.predict(face)[0]
        emotion = emotion_labels[np.argmax(prediction)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if emotion != last_spoken_emotion or (time.time() - last_spoken_time) > 5:
            engine.say(f"You look {emotion}")
            engine.runAndWait()
            last_spoken_emotion = emotion
            last_spoken_time = time.time()

    # Draw countdown timer
    cv2.putText(frame, f"Time Left: {remaining_time}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Emotion Recognition with Voice & Timer', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(" Exiting early by user.")
        break

cap.release()
cv2.destroyAllWindows()
