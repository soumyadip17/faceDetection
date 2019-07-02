import cv2
import os
import numpy as np
from PIL import Image
import face
import pickle
base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "images")
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
trained_images = []
label_images = []
dict ={}
count = 0
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('jpg'):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            if not label in dict:
                dict[label] = count
                count += 1
            id = dict[label]
            gray_image = Image.open(path).convert("L")
            size = (550,550)
            final_image = gray_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, 'uint8')
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                trained_images.append(roi)
                label_images.append(id)

with open("labels.pickle", 'wb') as f:
    pickle.dump(dict, f)

recognizer.train(trained_images, np.array(label_images))
recognizer.save('trainner.yml')