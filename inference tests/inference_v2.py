import cv2
import numpy as np
import tensorflow as tf
import json

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter("/Users/tanmay/Downloads/weights/cfg_2/fold-1/model.tflite")
interpreter.allocate_tensors()

REQUIRED_SIGNATURE = "serving_default"
REQUIRED_OUTPUT = "outputs"

# Load the character map
with open("/Users/tanmay/Downloads/weights/cfg_2/fold-1/inference_args.json", "r") as f:
    character_map = json.load(f)
# Create reverse map, handling list values
rev_character_map = {}
for i, j in character_map.items():
    if isinstance(j, list):
        for value in j:
            rev_character_map[value] = i
    else:
        rev_character_map[j] = i


# Check for the required signature
found_signatures = list(interpreter.get_signature_list().keys())
if REQUIRED_SIGNATURE not in found_signatures:
    raise Exception('Required input signature not found.')

# Get the signature runner for inference
prediction_fn = interpreter.get_signature_runner(REQUIRED_SIGNATURE)

# Initialize webcam feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set the desired frame size (for example, 224x224 if that is the model's input size)
frame_width, frame_height = 224, 224  # Adjust these dimensions based on your model

# Loop to continuously capture frames and run inference
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Resize the frame to match model input size
    resized_frame = cv2.resize(frame, (frame_width, frame_height))

    # Normalize the pixel values to [0, 1] if necessary (depending on your model's training preprocessing)
    normalized_frame = resized_frame.astype(np.float32) / 255.0

    # Add batch dimension: (1, height, width, channels)
    input_data = np.expand_dims(normalized_frame, axis=0)

    # Run inference
    output = prediction_fn(inputs=input_data)

    # Get the predicted character
    predicted_indices = np.argmax(output[REQUIRED_OUTPUT], axis=1)
    prediction_str = "".join([rev_character_map.get(idx, "") for idx in predicted_indices])

    # Display the predicted character on the frame
    cv2.putText(frame, f"Prediction: {prediction_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("ASL Fingerspelling Recognition", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
