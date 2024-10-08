import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="/Users/tanmay/Downloads/model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function for model inference
def predict(image):
    input_data = np.expand_dims(image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Function to process the webcam frame and perform prediction
def run_real_time_inference():
    cap = cv2.VideoCapture(0)  # Capture video from webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Check input shape, typically (1, height, width, channels)
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        # Resize frame to match model input size
        input_frame = cv2.resize(frame, (width, height))  

        # If model expects single channel (grayscale), convert frame
        if input_details[0]['shape'][3] == 1:
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
            input_frame = np.expand_dims(input_frame, axis=-1)  # Add channel dimension back
        
        input_frame = input_frame / 255.0  # Normalize if needed
        
        # Run prediction
        output = predict(input_frame)
        
        # Postprocess and display results
        prediction_str = np.argmax(output, axis=1)
        cv2.putText(frame, f'Prediction: {prediction_str}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Real-time Inference', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

run_real_time_inference()