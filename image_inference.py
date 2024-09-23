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

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def process_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe to extract landmarks
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        print("No hand detected in the image.")
        return None

    # Extract landmarks
    landmarks = []
    for hand_landmarks in results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])

    return landmarks

def test_model_with_landmarks(landmark_input):
    input_data = np.array(landmark_input).flatten().astype(np.float32)
    expected_features = inference_args.get('input_shape', [1, 390])[1]

    # Adjust landmarks to fit the input shape
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

def predict_from_image(image_path):
    landmarks = process_image(image_path)
    if landmarks is None:
        return

    # Get top 3 predictions from the model
    top_predictions = test_model_with_landmarks(landmarks)

    # Print the results
    print(f"Top 3 predictions for {image_path}:")
    for i, (pred_index, prob) in enumerate(top_predictions):
        print(f"Pred {i+1}: Class {pred_index} with confidence {prob:.2f}")

if __name__ == '__main__':
    # Example usage: Pass an image file path to the predict_from_image function
    image_file = '/Users/tanmay/Downloads/AdobeStock_314735118.jpeg'
    predict_from_image(image_file)
