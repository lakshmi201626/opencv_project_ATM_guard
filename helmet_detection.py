import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

face_cascade=cv2.CascadeClassifier(r'D:\DL_Projects\Project-DL\haarcascade_frontalface_default.xml')
face_cascade

import tensorflow
from tensorflow import keras

model=keras.models.load_model(r'D:\DL_Projects\Project-DL\DLoroject_model_new.h5')

video=cv2.VideoCapture(0)
while True:
    success,frame=video.read()
    faces=face_cascade.detectMultiScale(frame,minNeighbors=6)


    for (x,y,w,h) in faces:
        face_roi=frame[y-70:y+h+50,x-70:x+w+50]

        gray=cv2.cvtColor(face_roi,cv2.COLOR_BGR2GRAY)

           
        imgre = cv2.resize(gray, (150, 150))  # Resize to 150x150:

        img = imgre.reshape(1, 150, 150, 1)
        #print(img.shape)
             
         
        li=['without_helmet','with_helmet']

        predictions = model.predict(img)      #  prediction 
        print(predictions)
        ind=predictions.argmax(axis=1)
        predictions=li[ind.item()]
        print(predictions)
    
        cv2.rectangle(frame,(x-80,y-80),(x+w+50,y+h+60),(0,0,255),5)
#         if predictions == "without_helmet":  
#             message='u can continue'  
#             print(message)

#         elif predictions == "with_helmet": 
#             message = "Please remove the helmet."
#             print(message)

# #     # Draw bounding box (optional) and display the message on the frame
#         if message:
#                 # Code for drawing bounding box (if applicable)
#             cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        
    cv2.imshow('Webcam',frame)
    if cv2.waitKey(1) & 0XFF==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
