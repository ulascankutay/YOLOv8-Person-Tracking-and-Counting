from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import numpy as np
from sort import *
import threading
import queue

classNames = ["person"]
mask = cv2.imread("/home/ulascan/Desktop/Nesne Tanıma/Maske/maske5.jpg")

# Tracking
tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.5)

cap = cv2.VideoCapture("/home/ulascan/Desktop/Nesne Tanıma/Videolar/example_01.mp4")

# video kayıt için fourcc ve VideoWriter tanımlama
cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
success, img = cap.read()
#cv2.imwrite("/home/ulascan/Desktop/Nesne Tanıma/cepkamera.jpg", img)
size = list(img.shape)
del size[2]
size.reverse()
video = cv2.VideoWriter("SONUC4.mp4", cv2_fourcc, 60, size) #output video name, fourcc, fps, size

model = YOLO("yolov5nu.pt")

# Çizgi sınırları (giriş ve çıkış tespiti için)
limits = [1, 150 , 399, 150]
totalCount = []  # Toplam sayımı
enteredIDs = set()  # Giriş yapan ID'ler
exitedIDs = set()  # Çıkış yapan ID'ler
trackedObjects = {}  # ID'leri ve konumları takip et

frame_queue = queue.Queue(maxsize=10)
result_queue = queue.Queue()

def process_frame():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        imgRegion = cv2.bitwise_and(frame, mask)
        results = model(imgRegion, stream=True)

        detections = np.empty((0, 5))
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                if cls == 0:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)
        result_queue.put((frame, resultsTracker))

def main():
    num_threads = 4
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=process_frame)
        t.start()
        threads.append(t)

    prev_frame_time = 0
    while cap.isOpened():
        new_frame_time = time.time()
        success, img = cap.read()
        if not success:
            break

        frame_queue.put(img)
        
        if not result_queue.empty():
            frame, resultsTracker = result_queue.get()
            cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

            for result in resultsTracker:
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
                cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

                cx, cy = x1 + w // 2, y1 + h // 2
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                if id not in trackedObjects:
                    trackedObjects[id] = [cx, cy]
                else:
                    prev_x, prev_y = trackedObjects[id]
                    if limits[0] < cx < limits[2]:
                        if limits[1] - 50 < cy < limits[1] + 50:
                            if prev_y > limits[1] and cy <= limits[1]:
                                if id not in exitedIDs:
                                    exitedIDs.add(id)
                                    if id in enteredIDs:
                                        totalCount.remove(id)
                                    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
                            elif prev_y < limits[1] and cy >= limits[1]:
                                if id not in enteredIDs:
                                    enteredIDs.add(id)
                                    totalCount.append(id)
                                    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
                    trackedObjects[id] = [cx, cy]

            cv2.putText(frame, f'Giris: {len(enteredIDs)}', (15, 50), cv2.FONT_HERSHEY_PLAIN, 1, (50, 50, 255), 2)
            cv2.putText(frame, f'Cikis: {len(exitedIDs)}', (15, 125), cv2.FONT_HERSHEY_PLAIN, 1, (50, 50, 255), 2)

            video.write(frame)

            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            print("fps: ", fps)

            cv2.imshow("Image", frame)
            cv2.waitKey(1)

    for _ in range(num_threads):
        frame_queue.put(None)

    for t in threads:
        t.join()

    video.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
