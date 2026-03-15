import pandas as pd
import pickle
import streamlit as st

# Load model
model = pickle.load(open('car_price_model.pkl','rb'))
model_columns = pickle.load(open('model_columns.pkl','rb'))

cars_data = pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
    return car_name.split(' ')[0]

cars_data['name'] = cars_data['name'].apply(get_brand_name)

st.title("Car Price Prediction System")

name = st.selectbox("Car Brand", cars_data['name'].unique())
year = st.slider("Year", 1994, 2024)
km_driven = st.slider("KM Driven", 0, 200000)
fuel = st.selectbox("Fuel Type", cars_data['fuel'].unique())
seller_type = st.selectbox("Seller Type", cars_data['seller_type'].unique())
transmission = st.selectbox("Transmission", cars_data['transmission'].unique())
owner = st.selectbox("Owner", cars_data['owner'].unique())
mileage = st.slider("Mileage", 10, 40)
engine = st.slider("Engine CC", 700, 5000)
max_power = st.slider("Max Power", 0, 200)
seats = st.slider("Seats", 2, 10)

if st.button("Predict Price"):

    input_data = pd.DataFrame(
        [[name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats]],
        columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats']
    )

    input_data = pd.get_dummies(input_data)

    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_data)

    st.success(f"Estimated Price: ₹ {int(prediction[0]):,}")