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

def test_model_with_landmarks(landmark_input):
    input_data = np.array(landmark_input).flatten().astype(np.float32)
    expected_features = inference_args.get('input_shape', [1, 390])[1]  # Adjust based on your model

    # Adjust landmarks
    if input_data.size < expected_features:
        input_data = np.concatenate([input_data, np.zeros(expected_features - input_data.size)], axis=0)
    else:
        input_data = input_data[:expected_features]

    input_data = input_data.reshape(1, expected_features).astype(np.float32)

    # Set the tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probabilities = output_data[0]  # Assuming output is (1, num_classes)

    # Get top 3 predictions
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3_probs = probabilities[top_3_indices]

    return list(zip(top_3_indices, top_3_probs))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])

            # Predict using webcam input
            top_predictions = test_model_with_landmarks(landmarks)

            # Log the input and predictions for debugging
            print("Input Landmarks:", landmarks)
            #print("Top Predictions:", top_predictions)

            # Display top predictions
            """ for i, (pred_index, prob) in enumerate(top_predictions):
                cv2.putText(frame, f'Pred {i+1}: Class {pred_index} ({prob:.2f})', 
                            (10, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2) """

    cv2.imshow('ASL Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
hands.close()