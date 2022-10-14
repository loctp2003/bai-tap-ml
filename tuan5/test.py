import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
a = st.number_input('Nhap a:')
b = st.number_input('Nhap b:')
btn_giai = st.button('Giai')
if "btn_giai_state" not in st.session_state:
    st.session_state.btn_giai = False
if btn_giai or st.session_state.btn_giai_state: 
    st.session_state.btn_giai = True
    ax = plt.axes(projection="3d")
    x = np.linspace(-2, 2, 11)
    y = np.linspace(-2, 2, 11)
    X, Y = np.meshgrid(x,y)
    Z = X**a + Y**b
    ax.plot_wireframe(X, Y, Z)
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    st.plotly_chart()