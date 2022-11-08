from tkinter import Image
import tensorflow as tf
from tensorflow import keras 
import numpy as np
import cv2
import joblib
import streamlit as st
from PIL import Image as img  
import pickle as pkl  

btn_giai = st.button('Giai')
if btn_giai not in st.session_state:
    st.session_state.btn_giai = False
if btn_giai or st.session_state.btn_giai:
    pickle_in1 = open('knn_mnist.pkl', 'rb')  
    classifier1 = pkl.load(pickle_in1)  
    mnist = keras.datasets.mnist 
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data() 
    index = np.random.randint(0, 9999, 100)
    sample = np.zeros((100,28,28), np.uint8)
    for i in range(0, 100):
        sample[i] = X_test[index[i]]
        # 784 = 28x28
    RESHAPED = 784
    sample = sample.reshape(100, RESHAPED) 
    knn = joblib.load(classifier1)
    predicted = knn.predict(sample)
    k = 0
    for x in range(0, 10):
        for y in range(0, 10):
            st.write('%2d' % (predicted[k]), end='')
            k = k + 1
        st.write()

    digit = np.zeros((10*28,10*28), np.uint8)
    k = 0
    for x in range(0, 10):
        for y in range(0, 10):
            digit[x*28:(x+1)*28, y*28:(y+1)*28] = X_test[index[k]]
            k = k + 1
    cv2.imwrite('digit.jpg', digit)
    image = Image.open('digit.jpg')
    st.image(image,width=300)
     

        
   


