import pickle
import cv2
import mediapipe as mp
import numpy as np
import os

class HandGestureRecognition:
    def __init__(self, model_path='./model.p', labels_file='labels_dict.txt', static_mode=False, confidence=0.9, use_camera=False):
        # Load model
        self.model_dict = pickle.load(open(model_path, 'rb'))
        self.model = self.model_dict['model']

        # ✅ Only open webcam if needed
        # self.cap = cv2.VideoCapture(0) if use_camera else None
        # self.cap = cv2.VideoCapture()
        # MediaPipe hands setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            min_detection_confidence=confidence
        )

        # Load labels
        self.labels_dict = self.load_labels_from_file(labels_file)

    def load_labels_from_file(self, labels_file):
        """
        Load label dictionary from a text file. The file should have lines in the format:
        index:label
        Example:
        0:A
        1:B
        2:L
        """
        labels_dict = {}
        try:
            with open(labels_file, 'r') as f:
                for line in f:
                    # Remove any surrounding whitespace and split by ':'
                    line = line.strip()
                    if ':' in line:
                        key, value = line.split(':')
                        labels_dict[int(key)] = value
            print(f"Labels dictionary loaded from {labels_file}")
        except Exception as e:
            print(f"Error loading labels from file: {e}")
        return labels_dict

    def process_frame(self, frame):
        # Convert the frame to RGB and process it with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        prev_value = None
        predicted_character=None
        

        if results.multi_hand_landmarks:
            all_x_coords = []
            all_y_coords = []
            data_combined = []

            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                data_aux = []
                x_ = []
                y_ = []

                # Extract x and y coordinates
                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    x_.append(x)
                    y_.append(y)

                # Store all x and y coordinates for bounding box calculation
                all_x_coords.extend(x_)
                all_y_coords.extend(y_)

                # Normalize landmark coordinates relative to the bounding box
                min_x, min_y = min(x_), min(y_)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x - min_x
                    y = hand_landmarks.landmark[i].y - min_y
                    data_aux.append(x)
                    data_aux.append(y)

                data_combined.extend(data_aux)

            # Ensure correct feature length for both hands
            if len(data_combined) > 88:
                data_combined = data_combined[:88]  # Trim extra features if more than 88
            elif len(data_combined) < 88:
                data_combined.extend([0] * (88 - len(data_combined)))  # Pad with zeros if less

            # Make a single bounding box for both hands
            x1 = int(min(all_x_coords) * frame.shape[1]) - 10
            y1 = int(min(all_y_coords) * frame.shape[0]) - 10
            x2 = int(max(all_x_coords) * frame.shape[1]) + 10
            y2 = int(max(all_y_coords) * frame.shape[0]) + 10

            # Draw a single rectangle around both hands
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Make prediction using the trained model
            prediction = self.model.predict([np.asarray(data_combined)])

            # Get the predicted character
            predicted_character = self.labels_dict.get(int(prediction[0]), "Unknown")
            if predicted_character != prev_value:
                # print(predicted_character)
                prev_value = predicted_character 

            # Display the predicted character
            cv2.putText(frame, predicted_character, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return frame,predicted_character


    def start(self):
        pred_sentence=[]
        prev_frame=None
        current_frame=None
        count=0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Unable to read from camera.")
                break

            frame,predicted_character = self.process_frame(frame)
            
            # logic 101 to count the same frame
            if predicted_character:
                current_frame=predicted_character 
                if prev_frame is None:
                    prev_frame = current_frame
                if current_frame==prev_frame and current_frame:
                    print("same")
                    count+=1
                    print(count)
                else:
                    count=0
                    print("different")
                if count>=12 and current_frame:
                    print("add ",current_frame)
                    
                    # logic 100 to append the charactors to make sentense
                    
                    if len(pred_sentence)==0:
                        pred_sentence.append(predicted_character)
                        print(pred_sentence)
                    elif len(pred_sentence)==1 and predicted_character!=pred_sentence[0]:
                        pred_sentence.append(predicted_character)
                        print(pred_sentence)
                    elif len(pred_sentence)>1 and predicted_character!=pred_sentence[-1] and predicted_character!=pred_sentence[-2]:
                        pred_sentence.append(predicted_character)
                        print(pred_sentence)
                # logic 100 end
                prev_frame=current_frame
            # logic 101 end

            # # logic 100 to append the charactors to make sentense
            # if predicted_character:
            #     if len(pred_sentence)==0:
            #         pred_sentence.append(predicted_character)
            #         print(pred_sentence)
            #     elif len(pred_sentence)==1 and predicted_character!=pred_sentence[0]:
            #         pred_sentence.append(predicted_character)
            #         print(pred_sentence)
            #     elif len(pred_sentence)>1 and predicted_character!=pred_sentence[-1] and predicted_character!=pred_sentence[-2]:
            #         pred_sentence.append(predicted_character)
            #         print(pred_sentence)
            # # logic 100 end

            # Show the resulting frame
            cv2.imshow('Hand Gesture Recognition', frame)
            
            # Exit condition
            if (cv2.waitKey(1) & 0xFF == ord('q')) or predicted_character =='done.':
                print(" ".join(pred_sentence))
                break
        
        # Release the camera and close the window
        self.cap.release()
        cv2.destroyAllWindows()
        return (" ".join(pred_sentence))


    def save_labels_from_folders(directory_path, filename='labels_dict.txt'):
        try:
            # Retrieve all folders (subdirectories) in the given directory
            folder_names = [f for f in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, f))]

            # Create a dictionary where the key is the folder name and the value is the prefix of the first image
            labels_dict = {}
            for folder in folder_names:
                folder_path = os.path.join(directory_path, folder)
                image_files = [img for img in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, img))]
                if image_files:
                    # Sort image files to ensure consistent order and pick the first one
                    image_files.sort()
                    # Extract the prefix before the underscore (_) in the first image file name
                    first_image = image_files[0].split('_')[0]
                    labels_dict[folder] = first_image

            # Save the labels to the file in the required format
            with open(filename, 'w') as f:
                for key, value in labels_dict.items():
                    f.write(f"{key}:{value}\n")

            print(f"Labels dictionary saved to {filename}")
            
            # Printing the labels dict for verification
            print(f"Labels dictionary: {labels_dict}")
            
            return labels_dict  # Returning the dictionary for further use

        except Exception as e:
            print(f"Error saving labels dictionary: {e}")

if __name__ == '__main__':
    # Instantiate the HandGestureRecognition class and start the recognition
    HandGestureRecognition.save_labels_from_folders(directory_path='./data', filename='labels_dict.txt')
    hand_gesture_recognition = HandGestureRecognition(model_path='./model.p', labels_file='./labels_dict.txt')
    hand_gesture_recognition.start()
