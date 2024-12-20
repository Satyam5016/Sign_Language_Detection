import os
import pickle
import mediapipe as mp
import cv2

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Mediapipe Hands object
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Path to dataset directory
DATA_DIR = './data'

# Lists to store data and labels
data = []
labels = []

# Iterate over each subdirectory in DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue  # Skip non-directories like .gitignore
    
    # Iterate over each image in the subdirectory
    for img_path in os.listdir(dir_path):
        data_aux = []
        x_ = []
        y_ = []

        # Read the image
        img = cv2.imread(os.path.join(dir_path, img_path))
        if img is None:
            print(f"Warning: Unable to read image {img_path}. Skipping...")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with Mediapipe Hands
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Append to data and labels
            data.append(data_aux)
            labels.append(dir_)

# Save the dataset to a pickle file
output_file = 'data.pickle'
with open(output_file, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset created successfully and saved to {output_file}")
