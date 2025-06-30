# Real-Time Facial Emotion Recognition System

This project implements a real-time facial emotion recognition system using a Convolutional Neural Network (CNN), OpenCV for video capture and face detection, and TensorFlow for model training and prediction. It classifies facial expressions into one of seven emotions: **Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**.

---

## Features

- Real-time webcam-based facial emotion recognition
- CNN trained on FER2013 dataset
- Voice feedback of predicted emotion using `pyttsx3`
- Automatic timeout after 1 minute with countdown display
- Emotion label overlay on webcam feed

---

## Model

- **Architecture:** Custom CNN trained using TensorFlow
- **Accuracy:** ~81% on test set
- **Input size:** 48x48 grayscale images

---

## Technologies Used

- Python 3.8+
- OpenCV
- TensorFlow / Keras
- NumPy, Pandas
- pyttsx3 (for voice feedback)


