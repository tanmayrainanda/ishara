import tensorflow as tf
import cv2
import numpy as np
import json

# Load your model (update the path to your saved model)
model = tf.keras.models.load_model('/Users/tanmay/Downloads/aslfr-fp16-192d-17l-CTCGreedy.tflite')

# Load character-to-prediction index mapping
with open('/Users/tanmay/Downloads/kaggle-asl-fingerspelling-1st-place-solution-0.0.1-alpha/datamount/character_to_prediction_index.json', 'r') as f:
    char_to_index = json.load(f)

# Parameters based on the training preprocessing
MAX_LEN = 224  # Example max sequence length, update if needed

# Define a function to filter NaNs (missing values)
def filter_nans_tf(x):
    return np.nan_to_num(x, nan=0.0)

# Function to check if the input is left-handed
def is_left_handed(x):
    lhand = x[:, :21]  # Assuming 21 points per hand (for keypoints)
    rhand = x[:, 21:]
    lhand_nans = np.sum(np.isnan(lhand))
    rhand_nans = np.sum(np.isnan(rhand))
    return lhand_nans < rhand_nans

# Function to flip the frame horizontally for left-handed gestures
def flip_lr(x):
    return np.flip(x, axis=1)

# Preprocess the frame as per the training setup
def preprocess_frame(frame):
    # Resize the frame to the input size of the model (update dimensions as needed)
    frame_resized = cv2.resize(frame, (MAX_LEN, MAX_LEN))  
    frame_resized = filter_nans_tf(frame_resized)  # Handle missing values

    if is_left_handed(frame_resized):
        frame_resized = flip_lr(frame_resized)  # Flip left-handed inputs

    # Normalize the frame (assuming mean and std are computed beforehand)
    frame_resized = frame_resized.astype('float32') / 255.0  # Normalization
    frame_expanded = np.expand_dims(frame_resized, axis=0)  # Add batch dimension
    return frame_expanded

# Start capturing from webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Perform inference
    predictions = model.predict(processed_frame)
    
    # Get the class with the highest confidence
    predicted_index = np.argmax(predictions)
    
    # Get the corresponding character from the JSON mapping
    predicted_character = char_to_index.get(str(predicted_index), "Unknown")

    # Display the result on the frame
    cv2.putText(frame, f'Predicted: {predicted_character}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Print the raw predictions for debugging
    print("Predictions:", predictions)

    # Display the frame
    cv2.imshow('ASL Translation', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
