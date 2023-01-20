import os
import cv2
import pickle
import numpy as np
import random

IMG_SIZE = 100

directory = 'letsss'
maskdirectory = 'segmentations'

types = ["Black_Footed_Albatross","Laysan_Albatross","Sooty_Albatross","Groove_Billed_Ani","Crested_Auklet","Least_Auklet","Parakeet_Auklet","Rhinoceros_Auklet","Brewer_Blackbird","Red_Winged_Blackbird","Rusty_Blackbird","Yellow_Headed_Blackbird","Bobolink","Indigo_Bunting","Lazuli_Bunting","Painted_Bunting","Cardinal","Spotted_Catbird","Gray_Catbird","Yellow_Breasted_Chat"]

training_data = []

for files in os.listdir(directory):
    for img in os.listdir(f"{directory}/{files}"):
            if img.endswith(".jpg"):
                    imgname = img.split("_0")[0]
                    maskname = img.replace(".jpg", ".png")
                    class_num = types.index(imgname)
                    mask = cv2.imread(f"{maskdirectory}/{files}/{maskname}",cv2.IMREAD_GRAYSCALE)
                    mask = np.array(mask, dtype=np.uint8)
                    path = os.path.join(f"{directory}/{files}/{img}")
                    imgarray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    imgarray = cv2.resize(imgarray, (IMG_SIZE, IMG_SIZE))
                    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
                    img = cv2.bitwise_and(imgarray, mask)#takes only 8 bit
                    imgarray = img.flatten()
                    training_data.append([imgarray,class_num])
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 10000)
y = np.array(y)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close           
