import cv2
import os

import numpy as np

# Load the training images and labels

# Path to the training images
train_path = "/home/dhanish/PycharmProjects/ANN_Network/faces"
result=[]
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Get the training data we previously made
folders = os.listdir(train_path)
print(folders)
current_id = 0
training_images = []
training_labels = []
for folder in folders:
    print("folder", folder)
    image_path = os.path.join(train_path, folder)
    # print the file names in the folder
    images = os.listdir(image_path)
    for image in images:
        print(image)
        if image.endswith(".jpg"):
            image_path = os.path.join(train_path, folder, image)
            print(image_path)
            image = cv2.imread(image_path)
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_array = np.array(grey, "uint8")
            print(image_array)
            print(current_id)
            training_images.append(image_array)
            training_labels.append(current_id)
    info ={"Name": folder, "ID": current_id}
    result.append(info)
    current_id += 1
print(result)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(training_images, np.array(training_labels))

# Load the test image and predict the label
test_image = cv2.imread("/home/dhanish/PycharmProjects/ANN_Network/test/4.jpg")
gray_test = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(gray_test, scaleFactor=1.05, minNeighbors=6, minSize=(30, 30))
for (x, y, w, h) in faces:
    face_region = gray_test[y:y + h, x:x + w]
# cv2.imshow("test image", face_region)
# cv2.waitKey(0)
label, confidence = model.predict(face_region)

# Print the predicted label and confidence
print("Predicted label:", label)
print("Predicted Name:", result[label]["Name"])
print("Confidence:", confidence)
camera = cv2.VideoCapture(0)
while True:
    ret, frame = camera.read()

    gray_test = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # print("Predicted label:", label)
    faces = face_classifier.detectMultiScale(gray_test, scaleFactor=1.05, minNeighbors=6, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face_region_gray = gray_test[y:y + h, x:x + w]
        face_region_color = frame[y:y + h, x:x + w]
        label, confidence = model.predict(face_region_gray)
        cv2.putText(frame, result[label]["Name"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y + 10), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Face', frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    camera.release()


