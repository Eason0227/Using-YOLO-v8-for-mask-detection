import cv2
import math
from ultralytics import YOLO

# start webcam
cap = cv2.VideoCapture(0) # 創建一個VideoCapture物件並將其設置為從默認相機（0）捕獲幀
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("C:/Users/assd4/OneDrive/桌面/face mask detection/best.pt")

# object classes
classNames =  ["without_mask", "with_mask", "mask_weared_incorrect"]

while True:
    success, img = cap.read() # 將幀傳遞給YOLO模型以進行物件檢測
    results = model(img, stream=True) # 物件檢測的結果存儲result

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam  提取檢測到的對象的邊界框座標，並使用在其周圍繪製一個矩形。
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()