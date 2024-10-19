import cv2
import numpy as np
import os
import csv
import time
import pickle

from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

video = cv2.VideoCapture(0)
faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

names_path = 'data/names.pkl'
face_data_path  = 'data/face_data.pkl'
with open(names_path, 'rb') as w:
    LABELS = pickle.load(w)

with open(face_data_path, 'rb') as f:
    FACES = pickle.load(f)

if FACES.shape[0] > len(LABELS):
    # Truncate the face data to match the number of labels
    FACES = FACES[:len(LABELS)]
elif len(LABELS) > FACES.shape[0]:
    # Truncate the labels to match the number of face samples
    LABELS = LABELS[:FACES.shape[0]]


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgbackground = cv2.imread("bg.png")
COL_NAMES = ['NAME', 'TIME']
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        predicted_name = output[0]

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        attendance = [predicted_name, timestamp]

        exists = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-30), (x + w, y ), (50, 50, 255), -1)
        cv2.putText(frame, predicted_name,(x, y),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        attendance = [str(output[0]), str(timestamp)]
        # print(faces)
    imgbackground[162:162 + 480, 55:55 + 640] = frame

    cv2.imshow("frame", imgbackground)
    k = cv2.waitKey(1)
    if k == ord('o'):
        time.sleep(5)
        if exists:
            with open("Attendance/Attendance_" + date + ".csv") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Attendance/Attendance_" + time + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
