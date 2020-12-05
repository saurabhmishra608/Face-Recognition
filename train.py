import cv2
import numpy as np
from PIL import Image
import os

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    
    images = []
    labels = []
    count = 0
    for f in os.listdir(path):
        
        PATHS = os.listdir('dataset/'+ f)
        for image in PATHS:
            pth = 'dataset/' + f + '/' + image
            img = cv2.imread(pth, 0)
            
            img = np.array(img,'uint8')
            images.append(img)
            labels.append(count)
        count = count + 1
   
    labels = np.array(labels)           
    return images,labels        

faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
recognizer.write('trainer/trainer.yml')
print("Training finished...")