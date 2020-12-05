import numpy as np
import cv2
import time

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
cap = cv2.VideoCapture(0)
start = time.time()
while(cap.isOpened()):

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        cv2.putText(frame,'Face Detected',(0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        gray_roi = gray[y:y+h,x:x+w]
        color_roi = frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(gray_roi)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(color_roi,(ex,ey),(ex+ew,ey+eh),(0,0,255),3)
    cv2.imshow("result",frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()