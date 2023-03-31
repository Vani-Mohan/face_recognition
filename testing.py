import json

import cv2

model= cv2.face.LBPHFaceRecognizer_create()
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model.read("face_model.yml")

# To predict the image we need to load the image and convert it to gray scale
test_image = cv2.imread("Path to test image")
gray_test = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(gray_test, scaleFactor=1.05, minNeighbors=6, minSize=(30, 30))
for (x, y, w, h) in faces:
    face_region = gray_test[y:y + h, x:x + w]
label, confidence = model.predict(face_region)


with open('face.json', 'r') as f:
    result = json.load(f)
print("Predicted label:", label)
print("Predicted Name:", result[label]["Name"])
print("Confidence:", confidence)

# To predict the image using webcam

camera = cv2.VideoCapture(0)
while True:
    ret, frame = camera.read()
    print(ret)

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
