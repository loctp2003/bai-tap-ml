import streamlit as st
a = st.number_input('Nhap a:')
b = st.number_input('Nhap b:')
btn_giai = st.button('Giai')

if "btn_giai_state" not in st.session_state:
    st.session_state.btn_giai = False
if btn_giai or st.session_state.btn_giai_state: 
    st.session_state.btn_giai = True
    c = a + b
    ket_qua = "Nghiem c = %.2f" % c
    st.write(ket_qua)
