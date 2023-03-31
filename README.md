# face_recognition

This is a face recognition project that uses OpenCV and machine learning algorithms to detect and recognize faces in images and video streams. The project includes three Python scripts: one to create a dataset for face recognition, one to train a face recognition model on a dataset of faces, and a real-time face recognition application that can detect and recognize faces in video streams or images. 




# Installation

To use this project, you'll need to install OpenCV and Python 3 on your system. You can install OpenCV using pip:

```
pip install opencv-contrib-python

```

You'll also need to install the following Python packages:

```
pip install numpy

```

# Usage

To use the face recognition script, you'll need to prepare a dataset of faces that the model can be trained on. The dataset should be organized in the following format:

```
dataset/
    person1/
        image1.jpg
        image2.jpg
        ...
    person2/
        image1.jpg
        image2.jpg
        ...
    ...
```
You can create your own dataset using the file "create_dataset.py". The script will automatically create a folder with your name and capture 50 photos of you for the dataset.



Once you've prepared your dataset, you can train the face recognition model using the file "training.py"

This will generate a trained face recognition model and save it to the directory.

To run the real-time face recognition application, you can use the file "test.py"


This will open a video stream and display recognized faces in real-time.






