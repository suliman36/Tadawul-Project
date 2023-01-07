import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st




st.markdown("<h1 style='text-align: center; color: #505050	;'>Stock Trend Prediction</h1>", unsafe_allow_html=True)
user_input = st.text_input('Enter Stock Trading Name' )
df = pd.read_csv(r'C:\Users\sily\Downloads\LSTM\Tadawul\Tadawul\Tadawul_stocks_Prepro.csv')
df['date'] = pd.to_datetime(df.date, format='%m/%d/%Y')
df = df[df['trading_name '] == user_input]

#Describing Data
st.markdown("<h1 style='text-align: center; color: #282828	;font-size: 27px;'>Data from 2000 - 2019</h1>", unsafe_allow_html=True)
#st.subheader('Data from 2000 - 2019')
st.write(df[["high","low" , "open" , "close"  ,"value_traded" ,"volume_traded " ]].describe(include="all")) 

#Visualizations
st.markdown("<h1 style='text-align: center; color: #282828	;font-size: 27px;'>Closing Price vs Time Chart</h1>", unsafe_allow_html=True)

#st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (14,8))
plt.plot(  df.date , df.close)
plt.xlabel('Time' , fontsize=15)
plt.ylabel('Price' , fontsize=15)
st.pyplot(fig)


st.markdown("<h1 style='text-align: center; color: #282828	;font-size: 27px;'>Closing Price vs Time Chart with 100 days Moving Average</h1>", unsafe_allow_html=True)
#st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.close.rolling(50).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(df.date, ma100 , 'b' , label= 'Moving Average - 100 Days')
plt.plot(df.date , df.close , 'r' , label = 'Original Price' )
plt.xlabel('Time' , fontsize=15)
plt.ylabel('Price' , fontsize=15)
plt.legend()
st.pyplot(fig)

st.markdown("<h1 style='text-align: center; color: #282828	;font-size: 27px;'>Closing Price vs Time Chart with 100 & 200 days Moving Averages</h1>", unsafe_allow_html=True)
#st.subheader('Closing Price vs Time Chart with 100MA VS 200MA')
ma100 = df.close.rolling(50).mean()
ma200 = df.close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(df.date,ma100 , 'b' , label = 'Moving Average - 100 Days')
plt.plot(df.date,ma200 , 'g' , label = 'Moving Average - 200 Days')
plt.plot(df.date,df.close , 'r' ,label = 'Original Price')
plt.xlabel('Time' , fontsize=15)
plt.ylabel('Price' , fontsize=15)
plt.legend()
st.pyplot(fig)

#Splitting Data into Training and Testing

data_training = pd.DataFrame(df['close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['close'][int(len(df)*0.7) : int(len(df))])
scaler = MinMaxScaler(feature_range = (0,1))
data_training_array = scaler.fit_transform(data_training)



#Load keras model

model = load_model('Tadawul_keras_model.h5')

#Testing Model

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100 : i])
    y_test.append(input_data[i,0])

x_test , y_test = np.array(x_test) , np.array(y_test)
y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Graph
st.markdown("<h1 style='text-align: center; color: #282828	;font-size: 27px;'>Predictions vs Original</h1>", unsafe_allow_html=True)
#st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test , 'r' , label = 'Original Price')
plt.plot(y_predicted , 'y', label = 'Predicted Price')
plt.xlabel('Time' , fontsize=15)
plt.ylabel('Price' , fontsize=15)
plt.legend()
st.pyplot(fig2)
