import numpy as np
import cv2
faced=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')  
eye_cascadeglass = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
#train data for face detection and eye detection

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    face=faced.detectMultiScale(gray,1.1,5)
    for(x,y,w,h)  in face:
       cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
       roi_gray = gray[y:y+h, x:x+w]
       roi_color = frame[y:y+h, x:x+w]
       eyes = eye_cascade.detectMultiScale(roi_gray)
       eyesG = eye_cascadeglass.detectMultiScale(roi_gray)
       for (ex,ey,ew,eh) in eyes:
           cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),4)
       for (ex,ey,ew,eh) in eyesG:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),9)    

    cv2.imshow('frame for exit press q',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()