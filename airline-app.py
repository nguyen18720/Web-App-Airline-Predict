
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
#Airline Prediction App
""" )
st.sidebar.header('Input Feature')
st.sidebar.markdown("""
Example input
""")
#Collect user input feature into dataframe
upload_file = st.sidebar.file_uploader('Upload your input csv file',type=['csv'])
if upload_file is not None:
    input_df = pd.read_csv(upload_file)
else:
    def user_input_feature():
        Gender = st.sidebar.selectbox('Gender',('Female',"Male"))
        Customer_Type = st.sidebar.selectbox('Customer_type',('Loyal Customer','disloyal Customer'))
        Age = st.sidebar.slider('Age',7,85,27,1)
        Type_of_Travel = st.sidebar.selectbox('Type_of_Travel',('Personal Travel','Business Travel'))
        Class = st.sidebar.selectbox("Class",('Eco','Eco Plus','Business'))
        Flight_Distance = st.sidebar.slider('Flight Distance',0,7000,2000)
        Seat_comfort= st.sidebar.select_slider('Seat comfort',options=[1,2,3,4,5],value=(5))
        st.sidebar.write('My rating for this service is', Seat_comfort)
        Departure_Arrival_time_convenient = st.sidebar.select_slider('Departure/Arrival time convenient',options=[1,2,3,4,5],value=(5))
        st.sidebar.write('My rating for this service is', Departure_Arrival_time_convenient)
        Food_and_drink= st.sidebar.select_slider('Food and drink',options=[1,2,3,4,5],value=(5))
        st.sidebar.write('My rating for this service is', Food_and_drink)
        Gate_location= st.sidebar.select_slider('Gate location',options=[1,2,3,4,5],value=(5))
        st.sidebar.write('My rating for this service is', Gate_location)
        Inflight_wifi_service= st.sidebar.select_slider('Inflight_wifi_service',options=[1,2,3,4,5],value=(5))
        st.sidebar.write('My rating for this service is', Inflight_wifi_service)
        Inflight_entertainment= st.sidebar.select_slider('Inflight_entertainment',options=[1,2,3,4,5],value=(5))
        st.sidebar.write('My rating for this service is', Inflight_entertainment)
        Online_support= st.sidebar.select_slider('Online_support',options=[1,2,3,4,5],value=(5))
        st.sidebar.write('My rating for this service is', Online_support)
        Ease_of_Online_booking= st.sidebar.select_slider('Ease_of_Online_booking',options=[1,2,3,4,5],value=(5))
        st.sidebar.write('My rating for this service is', Ease_of_Online_booking)
        On_board_service= st.sidebar.select_slider('On_board_service',options=[1,2,3,4,5],value=(5))
        st.sidebar.write('My rating for this service is', On_board_service)
        Leg_room_service= st.sidebar.select_slider('Leg_room_service',options=[1,2,3,4,5],value=(5))
        st.sidebar.write('My rating for this service is', Leg_room_service)
        Baggage_handling= st.sidebar.select_slider('Baggage_handling',options=[1,2,3,4,5],value=(5))
        st.sidebar.write('My rating for this service is', Baggage_handling)
        Checkin_service= st.sidebar.select_slider('Checkin_service',options=[1,2,3,4,5],value=(5))
        st.sidebar.write('My rating for this service is', Checkin_service)
        Cleanliness= st.sidebar.select_slider('Cleanliness',options=[1,2,3,4,5],value=(5))
        st.sidebar.write('My rating for this service is', Cleanliness)
        Online_boarding= st.sidebar.select_slider('Online_boarding',options=[1,2,3,4,5],value=(5))
        st.sidebar.write('My rating for this service is', Online_boarding)
        Departure_Delay_in_Minutes = st.sidebar.slider('Departure_Delay_in_Minutes',0,1000,0)
        Arrival_Delay_in_Minutes = st.sidebar.slider('Arrival_Delay_in_Minutes',0,1000,0)
        data = {'Gender':Gender,
                'Customer_Type':Customer_Type,
                'Age':Age,
                'Type_of_Travel':Type_of_Travel,
                'Class':Class,
                'Flight_Distance':Flight_Distance,
                'Seat_comfort':Seat_comfort,
                'Departure/Arrival_time_convenient':Departure_Arrival_time_convenient,
                'Food_and_drink':Food_and_drink,
                'Gate_location':Gate_location,
                'Inflight_wifi_service':Inflight_wifi_service,
                'Inflight_entertainment':Inflight_entertainment,
                'Online_support':Online_support,
                'Ease_of_Online_booking':Ease_of_Online_booking,
                'On_board_service':On_board_service,
                'Leg_room_service':Leg_room_service,
                'Baggage_handling':Baggage_handling,
                'Checkin_service':Checkin_service,
                'Cleanliness':Cleanliness,
                'Online_boarding':Online_boarding,
                'Departure_Delay_in_Minutes':Departure_Delay_in_Minutes,
                'Arrival_Delay_in_Minutes':Arrival_Delay_in_Minutes}
        features = pd.DataFrame(data,index=[0])   
        return features
    input_df = user_input_feature() 
st.subheader('User Input Feature')
st.write(input_df) 

load_pre = pickle.load(open('Prepare_data.pkl','rb'))
set = load_pre.transform(input_df)
load_rdf = pickle.load(open('model-airline.pkl','rb'))
predict = load_rdf.predict(set)
predict_proba = load_rdf.predict_proba(set)

st.subheader("Prediction")
satis = np.array(['disatisfied','satisfied'])
st.write(satis[predict])

st.subheader("Prediction Probability")
st.write(predict_proba)
        
        
        