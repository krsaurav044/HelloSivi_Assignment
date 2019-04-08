# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 23:28:32 2019

@author: saurav
"""

from sklearn.cluster import KMeans
from collections import Counter
import cv2
import matplotlib.pyplot as plt

image= cv2.imread('test.jpg')
image_processing_size=(200,200)
if image_processing_size is not None:
    image = cv2.resize(image, image_processing_size, 
                            interpolation = cv2.INTER_AREA)
    
    #reshape the image to be a list of pixels
image = image.reshape((image.shape[0] * image.shape[1], 3))

    #cluster and assign labels to the pixels 
clt = KMeans(n_clusters = 100)
labels = clt.fit_predict(image)

    #count labels to find most popular
label_counts = Counter(labels)

    #subset out most popular centroid
dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
dominant_color=dominant_color.reshape(3,1)
labels=labels.reshape(200,200)
plt.imshow(labels)
labels=labels.reshape(200,200,1)
import scipy
scipy.misc.imsave(labels,'result1.jpg')