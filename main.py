from ultralytics import YOLO

import cv2
import cvzone

model=YOLO("yolov10n.pt")

# results = model('image2.png')
# results[0].show() 
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.5
font_thickness = 2
text_color = (255, 255, 255)  # White color for the text
rectangle_color = (0, 0, 0)
# print(results)
cap = cv2.VideoCapture(0)
while True:
    ret,image = cap.read()
    results = model(image)
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1,y1,x2,y2 = box.xyxy[0].numpy().astype('int')
            confidence = box.conf[0].numpy().astype('int')*100
            class_detected_number = box.cls[0]
            class_detected_number = int(class_detected_number)
            class_detected_name = results[0].names[class_detected_number]


            cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),3)
            
            cv2.putText(image, f'{class_detected_name}', org=[x1 + 8, y1 - 12],fontFace=4,fontScale=1,color=(255,0,0))

            


    cv2.imshow('frame',image)
    cv2.waitKey(1)


