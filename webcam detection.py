import torch
import cv2

# Load the YOLOv5 model (using a small pre-trained model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Perform real-time object detection
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Perform inference on the frame
    results = model(frame)

    # Render the results (bounding boxes and labels) on the frame
    results.render()  # Draw bounding boxes and labels on the frame

    # Get the annotated image from the results
    annotated_frame = results.ims[0]  # Use 'ims' to get the image

    # Convert the annotated image from RGB to BGR format for OpenCV display
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Display the frame with detection
    cv2.imshow('YOLOv5 Live Detection', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
