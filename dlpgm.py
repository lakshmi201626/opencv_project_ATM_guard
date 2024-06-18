import cv2
import numpy as np
from PIL import Image
import tensorflow
from tensorflow import keras
import streamlit as st
import pyttsx3



primaryColor="#161616"
backgroundColor="#efd6d6"
secondaryBackgroundColor="#926e6e"
textColor="#000000"
font="serif"

def speak_message(value):
    txt_sp=pyttsx3.init()
    voices=txt_sp.getProperty('voices')
    txt_sp.setProperty('voice',voices[1].id)
    # Convert value to a string for speaking
    msg_to_speak = str(value)  
    txt_sp.say(msg_to_speak)
    txt_sp.runAndWait()  # Wait for speech to finish



def pred_msg(frame):
  model=keras.models.load_model(r'D:\DL_Projects\Project-DL\DLoroject_model_new.h5')


  gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  imgre = cv2.resize(gray, (150, 150))  # Resize to 150x150:
  img = imgre.reshape(1, 150, 150, 1)
  li=['without_helmet','with_helmet']

  predictions = model.predict(img)      #  prediction 
  ind=predictions.argmax(axis=1)
  predictions=li[ind.item()]

  if predictions == "without_helmet":  
    message='u can continue'  
    #st.write(mask_message)

  elif predictions == "with_helmet": 
     message = "Please remove the helmet."
      #st.write(mask_message)

  return message
   
   



def main():

  st.title("**ATM Guard**")
  st.header("_A system for Helmet Detection in ATMs_")
  st.markdown("Experience a new level of :violet[ATM security] with :blue[Deep Learning]. This intelligent system leverages :blue[helmet detection] to identify suspicious activity and protect your financial transactions.")
  

  st.subheader('Output')

    
  st.write('Select the output format: Image, Video, or Webcam')
  options = ["Select Here","Image","Webcam"]
  selected_option = st.selectbox("Select an Option", options=options)

  if selected_option=="Image":
    st.write('U selected Image option')
    uploaded_file=st.file_uploader('upload your file')    #for file uploader widget

    if uploaded_file is not None:
        
        prediction = st.button('Prediction')

        if prediction:
            #st.write(uploaded_file)
            img = uploaded_file.read()

            st.image(img,"Uploaded Image")
            
            newimg = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)  # decode the image from a byte array and convert it into a color image

            msg=pred_msg(newimg)

            if msg=="Please remove the helmet.":
               st.write('Person in the image is wearing Helmet')
            elif msg=="u can continue":
               st.write('The person in this image not wearing helmet')
                
    else:
        # User didn't upload an image
      st.success("Please upload an image before making a prediction.")
    

  elif selected_option=="Webcam":
    st.write("You selected:", selected_option)
    button=st.button('Click here to open your Webcam')
    
    if button:
       
      st.success('Your webcam will open within seconds. To close the Webcam please select the window that is presenting and press "q" from your keyboard')

      # Start video capture
      video = cv2.VideoCapture(0)

      while video.isOpened(): 
        success, image = video.read()

        # Convert image to RGB format (MediaPipe expects)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        msg=pred_msg(image)
      
        if msg:
          cv2.putText(image, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        #speak_message(msg)
      

        cv2.imshow('Webcam',image)
       
        if cv2.waitKey(1) & 0XFF==ord('q'):
          break
      
      video.release()
      cv2.destroyAllWindows()
    




main()
