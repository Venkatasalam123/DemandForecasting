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
model = joblib.load('final_model_contoso_modified.pkl')
label_encoders = joblib.load('label_encoders_contoso_modified.pkl')

folder = '../Data/Synthetic/'
df_state_metric = pd.read_csv(os.path.join(folder, 'state_metric.csv'))
df_monthwise_metric = pd.read_csv(os.path.join(folder, 'monthwise_metric.csv'))
df = pd.read_csv(os.path.join(folder, 'Sales.csv'))

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

    holidays_list = [
    "2023-01-01",
    "2023-01-26",
    "2023-03-08",
    "2023-04-07",
    "2023-04-14",
    "2023-04-22",
    "2023-05-01",
    "2023-05-05",
    "2023-06-29",
    "2023-08-15",
    "2023-09-28",
    "2023-10-02",
    "2023-10-24",
    "2023-11-12",
    "2023-12-25"]
    
    df_sales['is_holiday'] = 0
    df_sales.loc[df_sales[date_column].isin(holidays_list), 'is_holiday'] = 1
    
    return df_sales



# Streamlit input example
st.title("Demand Forecasting App")

State = st.selectbox("State", label_encoders['State'].classes_)
Category = st.selectbox("Product Category", label_encoders['Product Category'].classes_)
Promotion = st.selectbox("Promotion Type", label_encoders['Promotion Type'].classes_)

df_filtered = df[(df['State'] == State) & (df['Product Category'] == Category)][['Date', 'Sales Quantity']]

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.date.today(), min_value=datetime.date(2000, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=datetime.date.today() + datetime.timedelta(days=10), min_value=start_date)

if start_date and end_date:
    if end_date >= start_date:
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        input_data = pd.DataFrame({
            'State': [State] * len(date_range),
            'Product Category': [Category] * len(date_range),
            'Promotion Type': [Promotion] * len(date_range),
            'Date': date_range
        })

        input_data = pd.merge(input_data, df_state_metric, on = ['State'])


        for col in ['State', 'Product Category', 'Promotion Type']:
            input_data[col] = label_encoders[col].transform(input_data[col])

        input_data['Date'] = pd.to_datetime(input_data['Date'])
        input_data = extract_date_components(input_data, 'Date')
        input_data.drop(['Date'], axis = 1, inplace = True)

        input_data = pd.merge(input_data, df_monthwise_metric[['Month', 'Money Supply M0', 'Money Supply M1', 'Inflation Rate']], on = ['Month'])
        columns = ['State', 'Population', 'Average High Temp', 'Average Low Temp',
            'Rainfall', 'Money Supply M0', 'Money Supply M1', 'Inflation Rate',
            'Product Category', 'Promotion Type', 'Year', 'Quarter', 'Month', 'Week', 'WeekOfMonth',
            'Day', 'Is_Weekend', 'DayOfWeek', 'DayOfYear', 'is_start_of_month',
            'is_end_of_month', 'season', 'is_holiday']
        input_data = input_data[columns]


        st.write('Influencing Features used are Population, Average High Temp, Average Low Temp, Rainfall, Money Supply M0, Money Supply M1, Inflation Rate')
        unique_rows_df = input_data.copy()
        unique_rows_df['State'] = State
        unique_rows_df['Product Category'] = Category
        unique_rows_df = unique_rows_df[['State', 'Product Category', 'Promotion Type', 'Year', 'Month', 'Population', 'Average High Temp', 'Average Low Temp',
        'Rainfall', 'Money Supply M0', 'Money Supply M1', 'Inflation Rate']].drop_duplicates()
        unique_rows_df.reset_index(drop = True, inplace = True)
        st.write(unique_rows_df)

        st.write('Below table shows how much each feature is influencing Sales quantity')
        fi = model.feature_importances_
        df_fi = pd.DataFrame(columns, columns = ['Columns'])
        df_fi['Feature Importance'] = fi
        df_fi.sort_values('Feature Importance', ascending = False, inplace = True)
        df_fi['Feature Importance'] = round((df_fi['Feature Importance'] * 100), 1)
        df_fi['Feature Importance'] = df_fi['Feature Importance'].astype(str) + ' %'
        # df_fi.reset_index(drop = True, inplace = True)
        df_fi.set_index('Columns', inplace = True)
        st.write(df_fi.T)

        # if st.button("Predict"):
        predictions  = model.predict(input_data)
        result_df = pd.DataFrame({
            'Date': date_range,
            'Predicted Sales Quantity': predictions
        })

        result_df['Date'] = pd.to_datetime(result_df['Date'])
        df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])
        result_df = result_df.set_index('Date')
        df_filtered = df_filtered.set_index('Date')
        merged_df = df_filtered.join(result_df, how='outer') 
        st.write("## Prediction")
        st.line_chart(merged_df)

        

            
    else:
        st.error("End date must be on or after the start date.")
else:
    st.info("Please select both start and end dates.")