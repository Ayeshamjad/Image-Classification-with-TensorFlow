import numpy as np
from tensorflow.keras import datasets, layers, models
import cv2 as cv
import matplotlib.pyplot as plt

class_names= ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

model = models.load_model('image_classifier.model')
img = cv.imread('deer.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

prediction = model.predict(np.array([img])/255)
index = np.argmax(prediction)

print(f"Prediction is {class_names[index]}")