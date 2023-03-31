import cv2 as cv
import os
import numpy as np

# Load the training images and labels

# Path to the training images
train_path = "Path to the training images"

face_classifier = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Get the training data we previously made
folders = os.listdir(train_path)
current_id = 0
result=[]
training_images = []
training_labels = []

for folder in folders:
    image_path = os.path.join(train_path, folder)
    images = os.listdir(image_path)
    for image in images:
        if image.endswith(".jpg"):
            image_path = os.path.join(train_path, folder, image)
            image = cv.imread(image_path)
            grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image_array = np.array(grey, "uint8")
            training_images.append(image_array)
            training_labels.append(current_id)
    info ={"Name": folder, "ID": current_id}
    result.append(info)
    current_id += 1

model = cv.face.LBPHFaceRecognizer_create()
model.train(training_images, np.array(training_labels))

# Load the test image and predict the label

camera = cv.VideoCapture(0)
while True:
    ret, frame = camera.read()
    if frame is None:
        continue

    gray_test = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # print("Predicted label:", label)
    faces = face_classifier.detectMultiScale(gray_test, scaleFactor=1.05, minNeighbors=6, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face_region_gray = gray_test[y:y + h, x:x + w]
        face_region_color = frame[y:y + h, x:x + w]
        label, confidence = model.predict(face_region_gray)
        cv.putText(frame, result[label]["Name"], (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.rectangle(frame, (x, y + 10), (x + w, y + h), (0, 255, 0), 2)
        cv.imshow('Face', frame)
        cv.waitKey(1)
cv.destroyAllWindows()
camera.release()


