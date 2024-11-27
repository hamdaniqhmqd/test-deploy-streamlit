import pickle
import streamlit as st
import pandas as pd
import numpy as np

model = pickle.load(open('model_prediksi_harga_mobil.sav', 'rb'))

st.title("Prediksi Harga Mobil")

st.header("Dataset")

df1 = pd.read_csv('CarPrice_Assignment.csv')
st.dataframe(df1)

st.write("Grafik Highway-mpg")
chart_highwaympg = pd.DataFrame(df1, columns=['horsepower'])
st.line_chart(chart_highwaympg)

st.write("Grafik Curb Weight")
chart_curbweight = pd.DataFrame(df1, columns=['horsepower'])
st.line_chart(chart_curbweight)

st.write("Grafik Horsepower")
chart_horsepower = pd.DataFrame(df1, columns=['horsepower'])
st.line_chart(chart_horsepower)

# input nilai dari variable independent
highwaympg = st.number_input("Highway-mpg")
curbweight = st.number_input("Curb Weight")
horsepower = st.number_input("Horsepower")

if st.button("Prediksi"):
  car_prediction = model.predict([[highwaympg,curbweight,horsepower]])
  
  harga_mobil_str = np.array(car_prediction)
  harga_mobil_float = float(harga_mobil_str[0])
  
  harga_mobil_formatted = "{:,}".format(harga_mobil_float)
  st.write(f"Hasil Diprediksi: ${harga_mobil_formatted}")
  
grafik = st.selectbox("Pilih Grafik", [
  'wheelbase', 'curbweight', 'carheight', 
  'carwidth', 'carlength', 'horsepower', 
  'peakrpm', 'citympg', 'highwaympg', 'price'
  ])

if grafik:
  st.write("Grafik ", grafik)
  chart_grafik = pd.DataFrame(df1, columns=[grafik])
  st.line_chart(chart_grafik)
