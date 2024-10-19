import os
import pickle
import cv2
import numpy as np

video = cv2.VideoCapture(0)  # 0 for webcam

faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_data = []
i = 0
num_samples = 100

name = input("Enter your name: ")

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture video")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50))

        # Add the face data every 10th frame to avoid duplicates
        if len(face_data) < num_samples and i % 10 == 0:
            face_data.append(resized_img)
            cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

        i += 1

    print(str(len(face_data)))

    cv2.imshow("frame", frame)
    k = cv2.waitKey(1)
    if len(face_data) == num_samples:
        break

video.release()
cv2.destroyAllWindows()

# save faces in pickle file
face_data = np.array(face_data)
face_data = face_data.reshape(num_samples, -1)

if not os.path.exists('data'):
    os.makedirs('data')

# Handle names
names_path = 'data/names.pkl'
if not os.path.exists(names_path):
    names = [name] * num_samples
    with open(names_path, 'wb') as f:
        pickle.dump(names, f)
else:
    with open(names_path, 'rb') as f:
        names = pickle.load(f)
    names.extend([name] * num_samples)

    with open(names_path, 'wb') as f:
        pickle.dump(names, f)

# Handle face data
face_data_path = 'data/face_data.pkl'
if not os.path.exists(face_data_path):
    with open(face_data_path, 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open(face_data_path, 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, face_data, axis=0)

    with open(face_data_path, 'wb') as f:
        pickle.dump(faces, f)

print(f"Data saved for {name}!")
