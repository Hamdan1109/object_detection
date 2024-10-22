from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov10n.pt")

# Define font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.5
font_thickness = 2
text_color = (255, 255, 255)  # White color for the text

cap = cv2.VideoCapture(0)

# Bounding box area thresholds for distance-based decision
stop_threshold = 40000  # Approximate area when person is too close (< 2 meters)
go_threshold = 20000    # Approximate area when person is at a safe distance (> 2 meters)

while True:
    ret, image = cap.read()
    if not ret:
        break

    # Run detection on the image
    results = model(image)

    # Flags to determine what message to display
    display_message = "GO"  # Default to "GO" when no one is close

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
            width = x2 - x1
            height = y2 - y1
            area = width * height

            class_detected_number = int(box.cls[0])
            class_detected_name = results[0].names[class_detected_number]

            # Draw bounding box for all detected objects
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # Display object name on the image
            cv2.putText(image, class_detected_name, (x1 + 8, y1 - 12), font, 1, text_color, font_thickness)

            # If a person is detected, check distance based on bounding box area
            if class_detected_name == 'person':
                if area > stop_threshold:
                    display_message = "STOP"  # Close (less than 2 meters)
                elif area < go_threshold:
                    display_message = "GO"  # Far (greater than 2 meters)

    # Display the appropriate message ("STOP" or "GO") based on the distance
    cv2.putText(image, display_message, (50, 100), font, font_scale, (0, 255, 0) if display_message == "GO" else (0, 0, 255), font_thickness)

    # Show the image with detection
    cv2.imshow('frame', image)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAll
