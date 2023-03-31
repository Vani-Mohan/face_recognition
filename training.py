import json

import cv2
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

with open('face.json', 'w') as f:
    json.dump(result, f)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(training_images, np.array(training_labels))
model.save("face_model.yml")



