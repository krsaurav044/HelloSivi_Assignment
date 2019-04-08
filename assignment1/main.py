# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 23:47:11 2019

@author: saurav
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import cv2
import sys

import time
from sklearn.preprocessing import StandardScaler
#importing of image for training
img_dir = "D:\\ml\\projects\\human\\dataset\\face"

images=[]
name=[]
i=0
for filename in os.listdir(img_dir):
    img = cv2.imread(os.path.join(img_dir,filename))
    if img is not None:
        img=resize(img,(100,100,3), mode = 'constant')
        images.append(img)
        name.append(1)
        i += 1

img_dir1 = "D:\\ml\\projects\\human\\dataset\\other"

for filename1 in os.listdir(img_dir1):
    img = cv2.imread(os.path.join(img_dir1,filename1))
    if img is not None:
        img=resize(img,(100,100,3), mode = 'constant')
        images.append(img)
        name.append(0)
        i += 1

images=np.asarray(images)
name=np.asarray(name)
name=name.reshape(5637,1)


X_train, X_test, y_train, y_test = train_test_split(images, name, test_size=0.1, random_state=42)
#developing model for training using transfer learning
from keras_vggface.vggface import VGGFace
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential,Model

classifier = VGGFace(model='vgg16',weights='vggface', include_top=False, input_shape=(100, 100, 3), pooling='avg')

for layer in classifier.layers[:]:
    layer.trainable = False
last_layer = classifier.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)


predictions = Dense(1, activation="sigmoid")(x)

model_final = Model(input = classifier.input, output = predictions)
model_final.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history=model_final.fit(X_train,y_train,batch_size=128,nb_epoch=10)

#detecting face of a human in an image using opencv
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
frame = cv2.imread('D:\\ml\\projects\\human\\test6.jpg')

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(100,100)
    )
x=0
y=0
w=0
h=0
for (x, y, w, h) in faces:
    img=cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    im=img
cv2.imshow('Video', frame)
detected_face = frame[int(y):int(y+h), int(x):int(x+w)]
'''detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
#detected_face = cv2.resize(detected_face, (48, 48))
detected_face = detected_face.astype(np.float64)
st=StandardScaler()
img_pixels=st.fit_transform(detected_face)
img_pixels=img_pixels.reshape(1,48,48,1)'''
#printing the result
y_pred=model_final.predict(detected_face)
if(y_pred<0.5):
    print('human')
else:
    print('non-human')

plt.imshow(im)


















