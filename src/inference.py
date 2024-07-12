import onnxruntime as ort
import numpy as np
import cv2
import os

# Loading ONNX model
onnx_model_path = "../models/sky_segmentation_model_25_epochs.onnx"
session = ort.InferenceSession(onnx_model_path)

# Getting model input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Initializing OpenCV video capture
cap = cv2.VideoCapture(0)  # Open default camera (index 0)
if not cap.isOpened():
    print("Failed to open camera.")
    exit()


def preprocess(frame):
    resized_frame = cv2.resize(frame, (256, 256))
    resized_frame = resized_frame.astype(np.float32) / 255.0
    # Normalizing using the same mean and std as the training script
    resized_frame -= np.array([0.485, 0.456, 0.406])
    resized_frame /= np.array([0.229, 0.224, 0.225])
    # Changing data layout from HWC to CHW
    chw_frame = np.transpose(resized_frame, (2, 0, 1))
    # Adding batch dimension
    chw_frame = np.expand_dims(chw_frame, axis=0)
    return chw_frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_tensor = preprocess(frame)

    # Run inference
    outputs = session.run([output_name], {input_name: input_tensor})
    output = outputs[0].squeeze(0)  # Remove batch dimension

    # Postprocess the output
    sigmoid_output = 1 / (1 + np.exp(-output))
    binary_mask = (sigmoid_output > 0.5).astype(np.uint8) * 255
    binary_mask = cv2.resize(binary_mask[0], (frame.shape[1], frame.shape[0]))

    # Creating a red mask where the segmented areas are highlighted in red
    red_mask = np.zeros_like(frame)
    red_mask[:, :, 2] = binary_mask  # Only the red channel

    # Overlaying the red mask on the frame
    overlay = cv2.addWeighted(frame, 0.8, red_mask, 0.5, 0.0)

    # Displaying the result
    cv2.imshow("Sky Segmentation", overlay)
    if cv2.waitKey(1) >= 0:
        break

cap.release()
cv2.destroyAllWindows()