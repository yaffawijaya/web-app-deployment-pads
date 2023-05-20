import streamlit as st
import pandas as pd
import webbrowser

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()


with header:
    st.title('Web App Data Analysis - Group 1')
    st.text('''
            Author: Group 1 Members 
            ''')
with dataset:
    st.header('NYC Taxi Dataset')
    st.text('We are currently using the NYC Yellow Taxi 2019 Deceber Dataset from this link down below:')

    url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-12.parquet'

    if st.button('Download Dataset'):
      webbrowser.open_new_tab(url)
    
    # taxi_data = get_data('taxi-data.parquet')
    taxi_data = pd.read_parquet('taxi-data.parquet')
    st.write(taxi_data.head())

    st.subheader('Pick up location ID distribution on the NYC Taxi Dataset')
    pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts()).head(50)
    st.bar_chart(pulocation_dist)

with features:
    st.title('The Features')

    st.markdown('* **First features:** reason of the features')
    st.markdown('* **Second features:** reason of the features')

with model_training:
    st.header('Time to train the model')
    st.text('Lorem ipsum dolor amet mantap pisan euy')

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=2, max_value=10, value=3 ,step=1)

    n_estimators = sel_col.selectbox('How many trees should there be?', options=[3,4,5,6,7,8], index=0)

  
    input_feature = sel_col.selectbox('Which feature should be used as the input feature?',options=taxi_data.columns)

    regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    X = taxi_data[[input_feature]].values.reshape(-1,1)
    y = taxi_data['trip_distance'].values

    regr.fit(X,y)
    prediction = regr.predict(X)
    mae = mean_absolute_error(y, prediction)
    disp_col.subheader('MAE of the model is:')
    disp_col.write(mae)
    disp_col.subheader('MSE of the model is:')
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader('R squared score of the model is:')
    disp_col.write(r2_score(y, prediction))