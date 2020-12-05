import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
args = parser.parse_args()
path = os.path.join("dataset",args.name)
os.mkdir(path)

cap = cv2.VideoCapture(0)
count = 0
print(path)
while(cap.isOpened()):

    rec, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    for (x,y,w,h) in faces:

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)     
        
    
    cv2.imshow('video', frame)  
    k = cv2.waitKey(1) & 0xff
    if k== ord('s'):
        count += 1
        cv2.imwrite(path + '/'+ args.name + str(count) + ".jpg", gray[y:y+h,x:x+w])
    elif k == ord('q'):
        break

    

      

cap.release()
cv2.destroyAllWindows()