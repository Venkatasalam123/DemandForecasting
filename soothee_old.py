import pandas as pd
#Import Libraries
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import joblib
import streamlit as st
import datetime


# Load the model and label encoders
model = joblib.load('final_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

def extract_date_components(df_sales, date_column):
    # Convert 'Invoice date' column to datetime if it's not already
    df_sales.loc[:, date_column] = pd.to_datetime(df_sales[date_column])
    
    # Extract year, quarter, month, week, weekofmonth, day, and is_weekend
    df_sales.loc[:, 'Year'] = df_sales[date_column].dt.year
    df_sales.loc[:, 'Quarter'] = df_sales[date_column].dt.quarter
    df_sales.loc[:, 'Month'] = df_sales[date_column].dt.month
    df_sales.loc[:, 'Week'] = df_sales[date_column].dt.isocalendar().week
    df_sales.loc[:, 'WeekOfMonth'] = df_sales[date_column].dt.day // 7 + 1
    df_sales.loc[:, 'Day'] = df_sales[date_column].dt.day
    df_sales.loc[:, 'Is_Weekend'] = df_sales[date_column].dt.dayofweek // 5
    df_sales.loc[:, 'DayOfWeek'] = df_sales[date_column].dt.dayofweek
    df_sales.loc[:, 'DayOfYear'] = df_sales[date_column].dt.dayofyear
    df_sales['is_start_of_month'] = df_sales[date_column].dt.is_month_start.astype(int)
    df_sales['is_end_of_month'] = df_sales[date_column].dt.is_month_end.astype(int)
    # df_sales['Is_First_Half_Of_Month'] = df_sales['Invoice date'].dt.day <= 10
    # df_sales['Is_Mid_Month'] = (df_sales['Invoice date'].dt.day > 10) & (df_sales['Invoice date'].dt.day <= 20)
    # df_sales['Is_Second_Half_Of_Month'] = df_sales['Invoice date'].dt.day > 20
    

    def get_season(month):
        if 3 <= month <= 5:
            return 1 #'Spring'
        elif 6 <= month <= 8:
            return 2 #'Summer'
        elif 9 <= month <= 11:
            return 3 #'Autumn'
        else:
            return 4 #'Winter'

    df_sales['season'] = df_sales[date_column].dt.month.apply(get_season)
    df_sales.sort_values(by = date_column, ascending = True, inplace = True)
    # df_sales.drop([date_column], axis = 1, inplace = True)
    
    return df_sales


# Streamlit input example
st.title("Demand Forecasting App")

Continent = st.selectbox("Continent", label_encoders['Continent'].classes_)
ProductSubcategoryName = st.selectbox("Product Sub Category", label_encoders['ProductSubcategoryName'].classes_)
ProductCategoryName = st.selectbox("Product Category", label_encoders['ProductCategoryName'].classes_)
Country = st.selectbox("Country", label_encoders['Country'].classes_)

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.date.today(), min_value=datetime.date(2000, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=datetime.date.today() + datetime.timedelta(days=10), min_value=start_date)

if start_date and end_date:
    if end_date >= start_date:
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        input_data = pd.DataFrame({
            'Continent': [Continent] * len(date_range),
            'ProductSubcategoryName': [ProductSubcategoryName] * len(date_range),
            'ProductCategoryName': [ProductCategoryName] * len(date_range),
            'Country' : [Country] * len(date_range),
            'Date': date_range
        })

        for col in ['Continent', 'ProductSubcategoryName', 'ProductCategoryName', 'Country']:
            input_data[col] = label_encoders[col].transform(input_data[col])

        input_data['Date'] = pd.to_datetime(input_data['Date'])
        input_data = extract_date_components(input_data, 'Date')
        input_data.drop(['Date'], axis = 1, inplace = True)

        if st.button("Predict"):
            predictions  = model.predict(input_data)
            result_df = pd.DataFrame({
                'Date': date_range,
                'Prediction': predictions
            })
            st.line_chart(result_df.set_index('Date'))
            # st.write(result_df)
    else:
        st.error("End date must be on or after the start date.")
else:
    st.info("Please select both start and end dates.")