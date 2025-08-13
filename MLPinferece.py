import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import tensorflow as tf
from keras.models import load_model

class HandGestureRecognition:
    def __init__(self, model_path='MLP_Keras_model.h5', label_encoder_path='label_encoder.pkl'):
        # Load Keras model
        self.model = load_model(model_path)
        print(f"Model loaded from {model_path}")

        # Load label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print(f"Label encoder loaded from {label_encoder_path}")

        # Initialize MediaPipe hands and webcam
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.70)

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        prev_value = None
        predicted_character = None
        print("MLP step1")
        if results.multi_hand_landmarks:
            all_x_coords = []
            all_y_coords = []
            data_combined = []
            print("MLP step1.2")
            for hand_landmarks in results.multi_hand_landmarks:
                print("MLP step1.3")
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                x_ = [landmark.x for landmark in hand_landmarks.landmark]
                y_ = [landmark.y for landmark in hand_landmarks.landmark]

                all_x_coords.extend(x_)
                all_y_coords.extend(y_)

                min_x, min_y = min(x_), min(y_)
                for landmark in hand_landmarks.landmark:
                    data_combined.append(landmark.x - min_x)
                    data_combined.append(landmark.y - min_y)
            print("MLP step2")
            # Pad or trim to 84 features
            if len(data_combined) > 84:
                data_combined = data_combined[:84]
            elif len(data_combined) < 84:
                data_combined.extend([0] * (84 - len(data_combined)))

            # Draw bounding box
            x1 = int(min(all_x_coords) * frame.shape[1]) - 10
            y1 = int(min(all_y_coords) * frame.shape[0]) - 10
            x2 = int(max(all_x_coords) * frame.shape[1]) + 10
            y2 = int(max(all_y_coords) * frame.shape[0]) + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            print("MLP step3")
            # Predict using Keras model
            prediction_prob = self.model.predict(np.array([data_combined]), verbose=0)
            print("MLP step4")
            prediction_class = np.argmax(prediction_prob)
            print("MLP step5")
            predicted_character = self.label_encoder.inverse_transform([prediction_class])[0]
            print("MLP",predicted_character)
            # data_directory = 'data'  # Replace with your actual path
            label_dict={
                '0': 'A', '1': 'B', '10': 'H', '11': 'I', '12': 'J', '13': 'K',
                '14': 'L', '15': 'M', '16': 'N', '17': 'O', '18': 'P',
                '4': 'E', '5': 'C', '6': 'D', '7': 'E', '8': 'F', '9': 'G', '19':'Me','20':'You'
            }

            label_dict[None]="unknown"

            predicted_character=label_dict[predicted_character]
            cv2.putText(frame, predicted_character, (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return frame, predicted_character

    def get_label_mapping(self,data_dir):
        label_map = {}

        # Loop through each folder in the data directory
        for folder in sorted(os.listdir(data_dir)):
            folder_path = os.path.join(data_dir, folder)

            if os.path.isdir(folder_path):
                # Get first file in the folder to extract the label name
                files = os.listdir(folder_path)
                if files:
                    first_file = files[0]
                    label_name = first_file.split('_')[0]
                    label_map[folder] = label_name

        return label_map


    def start(self):
        pred_sentence = []
        prev_frame = None
        current_frame = None
        count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Unable to read from camera.")
                break

            frame, predicted_character = self.process_frame(frame)

            if predicted_character:
                current_frame = predicted_character
                if prev_frame is None:
                    prev_frame = current_frame

                if current_frame == prev_frame:
                    count += 1
                else:
                    count = 0

                if count >= 12:
                    if len(pred_sentence) == 0 or (
                        len(pred_sentence) == 1 and predicted_character != pred_sentence[0]) or (
                        len(pred_sentence) > 1 and predicted_character != pred_sentence[-1] and predicted_character != pred_sentence[-2]):
                        pred_sentence.append(predicted_character)
                        print(pred_sentence)

                prev_frame = current_frame

            cv2.imshow('Hand Gesture Recognition', frame)

            if (cv2.waitKey(1) & 0xFF == ord('q')) or predicted_character == 'done.':
                print(" ".join(pred_sentence))
                break

        self.cap.release()
        cv2.destroyAllWindows()
        return " ".join(pred_sentence)


if __name__ == '__main__':
    hand_gesture_recognition = HandGestureRecognition(
        model_path='MLP_Keras_model.h5',
        label_encoder_path='label_encoder.pkl'
    )
    hand_gesture_recognition.start()