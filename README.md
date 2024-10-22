# Object Detection with YOLO

This code is a simple **object detection** and **distance-based decision-making** program that uses the **YOLOv8** model for detecting objects (specifically focusing on persons) via the webcam.

### Breakdown of the Code:

1. **Importing the Libraries**:
   - The `YOLO` class is imported from the `ultralytics` library, and `cv2` is imported from `OpenCV` for image processing and displaying.
   
2. **Loading the YOLO Model**:
   - The YOLOv8 model (`yolov10n.pt`) is loaded for real-time object detection, which detects objects from the webcam feed.

3. **Font and Camera Setup**:
   - A font is defined using OpenCV for overlaying text (such as "STOP" or "GO") onto the image.
   - The `cv2.VideoCapture(0)` captures video from the default webcam.

4. **Thresholds for Distance Calculation**:
   - Two thresholds (`stop_threshold` and `go_threshold`) are set based on the area of the bounding box. These values are used to estimate the distance of the detected person from the camera. If a person is too close (large bounding box area), it triggers a "STOP" message, and if the person is far enough (small bounding box area), it shows a "GO" message.

5. **Real-Time Detection Loop**:
   - Inside the loop:
     - Each frame is captured and fed to the YOLO model for detection.
     - The YOLO model returns detected objects and their bounding boxes.
     - Bounding boxes are drawn around each detected object using `cv2.rectangle`.
     - The detected object's class name (e.g., 'person') is displayed using `cv2.putText`.

6. **Distance-Based Decision**:
   - If the detected object is a person, the program checks the size of the bounding box (which roughly corresponds to how close the person is to the camera):
     - If the area is larger than `stop_threshold`, it assumes the person is too close and displays **"STOP"**.
     - If the area is smaller than `go_threshold`, it assumes the person is at a safe distance and displays **"GO"**.

7. **Displaying the Output**:
   - The message ("STOP" or "GO") is overlaid on the frame, and the processed frame is displayed using `cv2.imshow`.

8. **Exiting**:
   - The program continuously displays the processed video until the user presses the 'q' key, which exits the loop and closes the camera feed.

### Use Case:
- This code can be used in real-time object detection systems to monitor how close someone is to a camera and provide simple commands ("GO" or "STOP") based on their distance.

