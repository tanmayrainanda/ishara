import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
import json

# Load the TFLite model and inference args
interpreter = tf.lite.Interpreter(model_path='/Users/tanmay/Downloads/weights/cfg_2/fold-1/model.tflite')
interpreter.allocate_tensors()

with open('/Users/tanmay/Downloads/weights/cfg_2/fold-1/inference_args.json') as f:
    inference_args = json.load(f)


# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize MediaPipe hands and drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Capture from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Prepare landmarks for model input
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            
            # Convert to numpy array
            landmarks_array = np.array(landmarks).flatten().astype(np.float32)
            expected_features = inference_args.get('input_shape', [1, 390])[1]

            # Adjust landmarks
            if landmarks_array.size < expected_features:
                input_data = np.concatenate([landmarks_array, np.zeros(expected_features - landmarks_array.size)], axis=0)
            else:
                input_data = landmarks_array[:expected_features]

            # Ensure the final input is FLOAT32 and reshape for model input
            input_data = input_data.astype(np.float32).reshape(1, expected_features)

            # Set the tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # Run inference
            interpreter.invoke()

            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(output_data)

            # Display prediction
            cv2.putText(frame, f'Predicted: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('ASL Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
