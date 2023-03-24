import os
import cv2 as cv

photos_taken = 0  # Initialse the Number of photos taken

# Load the face classifier
face_classifier = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_classifier = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

# check if classifier file is loaded properly
if face_classifier.empty():
    raise IOError('Unable to load the face cascade classifier xml file')
if eye_classifier.empty():
    raise IOError('Unable to load the eye cascade classifier xml file')

dir_path = "Path to the directory where you want to save the images"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
    print(f"Directory '{dir_path}' created successfully.")
else:
    print(f"Directory '{dir_path}' already exists.")

name = input("Enter your name: ")
name = name.lower()

# create a directory with the name of the person
if not os.path.exists(os.path.join(dir_path, name)):
    os.mkdir(os.path.join(dir_path, name))

camera = cv.VideoCapture(0)
while (photos_taken < 50):     # Take 50 photos of each person

    ret, frame = camera.read()

    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=6, minSize=(30, 30))

    print("Number of faces detected: " + str(len(faces)))
    if len(faces) == 0:
        continue
    for (x, y, w, h) in faces:
        cv.rectangle(grey, (x, y), (x + w, y + h), (0, 0, 255), 2)
        face_region = grey[y:y + h, x:x + w]
        cv.imshow("face_region", grey)
        cv.waitKey(10)
        print("If you see a rectangle around your face, Press 's' to save the image. If not, press 'n' to continue")
        if (cv.waitKey(0) & 0xFF == ord('s')):
            img_name = str(photos_taken) + ".jpg"
            cv.imwrite(os.path.join(dir_path, name, img_name), face_region)
            photos_taken += 1
        else:
            continue


































