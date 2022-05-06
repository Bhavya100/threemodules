from calendar import c
import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import speech_recognition as sr
import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np
from gtts import gTTS 
import os 
from IPython.display import Audio
from googletrans import Translator
import tensorflow as tf




def main():

    st.title("Three apps")

    menu = ["Handwriting Detection", "OCR", "Speech to Text"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Handwriting Detection":
        st.subheader("Handwriting Detection")
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = tf.keras.utils.normalize(x_train, axis=1)
        x_test = tf.keras.utils.normalize(x_test, axis=1)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=224, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(units=30, activation=tf.nn.softmax))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=1)
        loss, accuracy = model.evaluate(x_test, y_test)
        st.info(loss)
        st.info(accuracy)
        model.save('digits.model')
        IMAGE_PATH = ('7.png')
        img = cv.imread(IMAGE_PATH)[:,:,0] 
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        st.info(f'The number is probably a: {np.argmax(prediction)}')
        plt.imshow(img[0])
        plt.show()

    elif choice == "OCR":
        st.subheader("OCR")
        IMAGE_PATH = ('alttext.jpg')
        translator = Translator()
        reader = easyocr.Reader(['en'])
        result = reader.readtext(IMAGE_PATH)
        st.info(result)
        top_left = tuple(result[0][0][0])
        bottom_right = tuple(result[0][0][2])
        text = result[0][1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.imread(IMAGE_PATH)
        spacer = 100
        for detection in result: 
            top_left = tuple(detection[0][0])
            bottom_right = tuple(detection[0][2])
            text = detection[1]
            img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
            img = cv2.putText(img,text,(20,spacer), font, 2,(255,255,255),5,cv2.LINE_AA)
            spacer+=70
    
            plt.imshow(img)
            plt.show()
            text_list = reader.readtext('alttext.jpg', detail = 0)
            st.info(text_list)
            text_comb=' '.join(text_list) 
            text_comb
            ta_tts=gTTS(text_comb)
            ta_tts.save('trans.mp3')
            Audio('trans.mp3' , autoplay=True)
            #text_es=translator.translate(text_comb, src='en' ,dest='es')
            #st.info(text_es.text)
            #ta_tts=gTTS(text_es.text)
            #ta_tts.save('transs.mp3')
            #Audio('transs.mp3' , autoplay=True)

    elif choice == "Speech to Text":
        st.subheader("Speech to Text")
        a = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Say Something : ")
            audio = a.listen(source)
            try:
                text = a.recognize_google(audio)
                st.info("You Said : {}".format(text))
        
            except:
                st.info("Sorry could not recognize")
        








if __name__ == "__main__":
    main()
