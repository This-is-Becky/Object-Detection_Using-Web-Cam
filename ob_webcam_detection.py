import cv2
from ultralytics import YOLO
import cvzone

#Test the download img to try how the YOLO works
model = YOLO("yolov8n.pt")
results= model("1.jpg", show=True)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Initialize webcam capture(0 is default build-in cam)
cap = cv2.VideoCapture(0)

#implement YOLO model
model = YOLO("yolov8n.pt")

#Create a loop to continuously capture frames from cam
while True:
    # Read a frame from the webcam
    success, img = cap.read()
    # Check if the frame was successfully captured
    if not success:
        print("Failed to capture frame")
        break
    results= model(img, stream=True)
    for x in results:
        boxes= x.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Extract label and confidence score
            label_id = int(box.cls[0])
            confidence = box.conf[0]
            label = model.names[label_id]  # Get the label name from the model

            # Draw bounding box and label with confidence score
            cvzone.cornerRect(img, (x1, y1, w, h))
            text = f"{label} {confidence:.2f}"
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the captured frame in a window
    cv2.imshow("Webcam", img)

    # Check for user input
    cv2.waitKey(1)
    # Exit the loop if 'closed' is clicked
    if cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()