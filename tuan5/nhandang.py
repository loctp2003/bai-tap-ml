import streamlit as st
image_file = st.file_uploader('Updload', type=['PNG','JPG','BMP'])

if image_file is not None:
    file_name = image_file.name
    file_name = "test/" + file_name
    st.write(file_name)
    st.image(file_name)