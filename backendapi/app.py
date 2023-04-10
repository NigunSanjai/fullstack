import csv
import chardet
import pandas as pd
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from sklearn.model_selection import train_test_split
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import pymongo
from flask import Flask, request, jsonify, session, make_response
from flask_restful import Api, Resource
from models import UserModel, db
from flask_cors import CORS, cross_origin
from flask_pymongo import PyMongo
from pymongo import MongoClient
from bson.json_util import dumps
from bson.objectid import ObjectId
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from time import sleep

import pymongo
from pylab import rcParams
import statsmodels.api as sm
from dateutil import parser
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn import metrics
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

app = Flask(__name__)
CORS(app)


client = MongoClient('mongodb://localhost:27017/')
db = client['mydatabase']
collection = db['users']

@app.route('/register', methods=['POST'])
def create_user():
    data = request.get_json()
    new_user = {
        'name': data['name'],
        'email': data['email'],
        'password': data['password'],
        'contact': data['contact']
    }
    if (collection.find_one({'email':data['email']})):
            print("validated")
            return jsonify(False), 401


    result = collection.insert_one(new_user)
    inserted_user = collection.find_one({'_id': ObjectId(result.inserted_id)})
    inserted_user['_id'] = str(inserted_user['_id'])
    return jsonify(inserted_user), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = collection.find_one({'email': data['email'], 'password': data['password']})
    if user:
        name = user.get('name')
        user['_id'] = str(user['_id'])
        if user.get('datasets'):
            print("yes");
            datasets = user.get('datasets')
            keys = list(datasets.keys())
            res = ','.join(keys)
            return jsonify({'result': True,'user_id': str(user['_id']),'name':name,'datasets':res})
        else:
            return jsonify({'result': True, 'user_id': str(user['_id']), 'name': name, 'datasets': user.get('datasets')})
    else:
        return jsonify({'result': False})


@app.route('/upload', methods=['POST'])
def upload():
    # Get the data from the request
    current_user = str(request.form.get('currentuser'))
    # title = request.form.get('title')
    # column_value = str(request.form.get('column_value'))
    # time_period = str(request.form.get('time_period'))
    # number_value = str(request.form.get('number_value'))
    file = request.files['file']
    filename = file.filename
    print(filename)
    print(current_user)
    df = pd.read_csv(file, encoding='latin-1')
    user_record = db.users.find_one({'_id': ObjectId(current_user)})
    if user_record:
        print("yes")
        csv_files = user_record.get('datasets', {})
        if filename in csv_files:
            # If the file already exists, you may choose to update it or return an error message
            return jsonify({'response': True }),200
        else:
            # Create a new collection with the file name
            csv_collection_name = filename
            csv_collection = db[csv_collection_name]
            csv_collection.insert_many(df.to_dict('records'))
            # Add the file name and collection name to the user record
            csv_files[filename] = csv_collection_name
            db.users.update_one({'_id': ObjectId(current_user)}, {'$set': {'datasets': csv_files}})
            return jsonify({'response': True}), 200
    else:
        return jsonify({'response':'User Not Found'}), 400
    print(column_value)
    print(time_period)
    print(number_value)



    return jsonify({'response': True}), 200


@app.route('/unique',methods=['POST'])
def get_prod_names():
    data = request.get_json()
    user = db.users.find_one({'_id': ObjectId(data['currentuser'])})
    filename=data['filename']
    collection = db[filename]

    # print the collections
    df = pd.DataFrame(list(collection.find()))
    product_names = df['PRODUCT_NAME'].unique()
    print(product_names)
    lists=[]
    for name in product_names:
        lists.append(str(name))
    return jsonify({'prodnames': lists}), 200

@app.route('/retrivefiles',methods=['POST'])
def get_filenames():
    data = request.get_json()
    print(data['currentuser'])
    user = db.users.find_one({'_id': ObjectId(data['currentuser'])})
    if user is None:
        return jsonify({'error': 'User not found'}), 404
    datasets = user.get('datasets')
    keys = list(datasets.keys())
    res = ','.join(keys)

    return jsonify({'filenames': res}), 200

@app.route('/predict',methods=['POST'])
def predict():
    data=request.get_json()
    current_user = data['currentuser']
    title = data['title']
    column_value = data['column_value']
    time_period = data['time_period']
    number_value = data['number_value']
    filename = data['filename']
    collection = db[filename]
    prod_name = data['prod_Name']

    df = pd.DataFrame(list(collection.find()))
    if (time_period == 'YEARLY'):
        freqs = 'Y'
    elif (time_period == 'MONTHLY'):
        freqs = 'M'
    elif (time_period == 'WEEKLY'):
        freqs = 'W'
    elif (time_period == "DAY_WISE"):
        freqs = 'D'
    df = df.loc[:, ['ORDER_DATE', 'PRODUCT_NAME', 'QUANTITY_ORDERED', 'MSRP', 'SALES',"PRICE_EACH"]]
    unique_products_count = str(df['PRODUCT_NAME'].nunique())
    print(unique_products_count)
    count = str(df['PRODUCT_NAME'].count())
    df['total_revenue'] = df['QUANTITY_ORDERED'] * df['PRICE_EACH']
    total_revenue = str(round(df['total_revenue'].sum(),2))
    filtered_df = df.loc[df['PRODUCT_NAME'] == prod_name]
    filtered_df = filtered_df.loc[filtered_df[column_value] > 0]
    filtered_df = filtered_df[['ORDER_DATE', column_value]]
    # filtered_df.columns = ['ds', 'y']
    df['ORDER_DATE'] = pd.to_datetime(df['ORDER_DATE'])
    filtered_df['ORDER_DATE'] = pd.to_datetime(filtered_df['ORDER_DATE'])
    start_date = df['ORDER_DATE'].min()
    # print(start_date)
    end_date = df['ORDER_DATE'].max()
    # print(end_date)
    all_dates = pd.date_range(start=start_date, end=end_date)
    all_data = pd.DataFrame({'ORDER_DATE': all_dates})
    all_data[column_value] = pd.np.nan
    merged_data = pd.merge(all_data, df, on='ORDER_DATE', how='left')

    df = df.sort_values(by="ORDER_DATE")
    df = df.loc[:, ['ORDER_DATE', 'PRODUCT_NAME', 'QUANTITY_ORDERED', 'MSRP', 'SALES', 'PRICE_EACH']]
    df.head()
    filtered_df = df.loc[df['PRODUCT_NAME'] == prod_name]
    filtered_df = filtered_df.loc[filtered_df[column_value] > 0]
    filtered_df = filtered_df[['ORDER_DATE', column_value]]
    filtered_df.head()
    start_date = filtered_df['ORDER_DATE'].min()
    end_date = filtered_df['ORDER_DATE'].max()
    all_dates = pd.date_range(start=start_date, end=end_date)
    all_data = pd.DataFrame({'ORDER_DATE': all_dates})
    all_data[column_value] = pd.np.nan
    merged_data = pd.merge(all_data, filtered_df, on='ORDER_DATE', how='left')
    merged_data = merged_data.drop(merged_data.columns[1], axis=1)
    merged_data = merged_data.sort_values('ORDER_DATE')
    print(merged_data.head())
    consolidated_data = merged_data.groupby('ORDER_DATE')[column_value+"_y"].sum().reset_index()
    # print(new_df)
    print(consolidated_data.head())
    print(consolidated_data.head())
    consolidated_data.columns = ['ds', 'y']
    print(consolidated_data.info())
    if (consolidated_data['y'] < 0).any():
        print('DataFrame contains negative values')
    else:
        print('DataFrame does not contain negative values')
    model = Prophet()
    model.fit(consolidated_data)

    # df_cv = cross_validation(model, initial='600 days', period='90 days', horizon='240 days')
    # df_p = performance_metrics(df_cv)
    # #
    # # Print evaluation metrics
    # print('MSE:', df_p['mse'].mean())
    # print('RMSE:', df_p['rmse'].mean())
    # print('MAE:', df_p['mae'].mean())
    # # # print('MAPE:', df_p['mape'].mean())



    future = model.make_future_dataframe(periods=len(all_dates), freq='D')
    forecast = model.predict(future)

    missing_dates_forecast = forecast.loc[:, ['ds', 'yhat']].rename(columns={'ds': 'ORDER_DATE', 'yhat': column_value})
    final_df = pd.concat([consolidated_data, missing_dates_forecast]).sort_values('ORDER_DATE')
    final_df.drop(['ds','y'],axis=1)
    if (final_df[column_value] < 0).any():
        final_df[column_value] = final_df[column_value].abs()
    # Filter out the training data from the forecast DataFrame
    last_date = final_df['ORDER_DATE'].max()
    future_dates = pd.date_range(start=last_date, periods=int(number_value), freq=freqs)
    # m = Prophet()
    # m.fit(final_df)
    # df_cv = cross_validation(model, initial='720 days', period='180 days', horizon='365 days')
    # df_p = performance_metrics(df_cv)
    # #
    # # Print evaluation metrics
    # print('MSE:', df_p['mse'].mean())
    # print('RMSE:', df_p['rmse'].mean())
    # print('MAE:', df_p['mae'].mean())
    # # print('MAPE:', df_p['mape'].mean())

    # Generate a forecast for the future date range
    future = pd.DataFrame({'ds': future_dates})
    forecast = model.predict(future)
    # Filter out the training data from the forecast DataFrame
    last_date = final_df['ORDER_DATE'].max()
    forecast_results = forecast.loc[forecast['ds'] > last_date, ['ds', 'yhat']]
    forecast_results.columns = ['ORDER_DATE', column_value]
    if (forecast_results[column_value] < 0).any():
        forecast_results[column_value] = forecast_results[column_value].abs()

    # Save the predicted results to a csv file
    org_name = current_user + title + column_value + time_period + number_value + prod_name + "long.csv"
    saved_name = current_user + title + column_value + time_period + number_value + prod_name + ".csv"
    fileName = str(saved_name)
    fileName1 = str(org_name)
    final_df.to_csv(org_name,index=False);
    forecast_results.to_csv(saved_name, index=False)

    # send to db
    df = pd.read_csv(saved_name)
    df1 = pd.read_csv(org_name)
    user_record = db.users.find_one({'_id': ObjectId(current_user)})
    powerbi = False
    if user_record:
        print("yes")
        csv_files = user_record.get('predicted', {})
        if saved_name not in db.list_collection_names():
        # Create a new collection with the file name
            csv_collection_name = saved_name
            csv_collection = db[csv_collection_name]
            csv_collection.insert_many(df.to_dict('records'))
        # Add the file name and collection name to the user record
            csv_files[fileName] = csv_collection_name
            db.users.update_one({'_id': ObjectId(current_user)}, {'$set': {'predicted': csv_files}})
        csv_files1 = user_record.get('original', {})
        if org_name not in db.list_collection_names():
        # Create a new collection with the file name
            csv_collection_name1 = org_name
            csv_collection1 = db[csv_collection_name1]
            csv_collection1.insert_many(df1.to_dict('records'))
        # Add the file name and collection name to the user record
            csv_files1[fileName1] = csv_collection_name1
            db.users.update_one({'_id': ObjectId(current_user)}, {'$set': {'original': csv_files1}})
    for i in range(50000000):
        pass  # Do nothing
    print('End')
    return jsonify({'response': True,"products":unique_products_count,"count":count,"revenue":total_revenue}), 200


@app.route('/visualise', methods=['POST'])
def post_request():
    data=request.get_json();
    current_user = data['currentuser']
    title = data['title']
    column_value = data['column_value']
    time_period = data['time_period']
    number_value = data['number_value']
    prod_name = data['prodname']
    name = current_user+title+column_value+time_period+number_value+prod_name+".csv"
    collection=db[name]
    documents = list(collection.find())

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(documents)
    print(df.head())
    #
    to_drop = ['_id']
    # # Check if columns to drop are present in dataframe
    df = df.drop(to_drop, axis=1)
    # Convert the DataFrame to a CSV file and return it as a Flask response
    csv_data = df.to_csv(index=False)
    # response = make_response(csv_data)
    # response.headers['Content-Disposition'] = 'attachment; filename=data.csv'
    # response.headers['Content-Type'] = 'text/csv'

    return jsonify({'file': csv_data})


@app.route('/getoriginal', methods=['POST'])
def org_request():
    data=request.get_json();
    current_user = data['currentuser']
    title = data['title']
    column_value = data['column_value']
    time_period = data['time_period']
    number_value = data['number_value']
    prod_name = data['prodname']
    name = current_user+title+column_value+time_period+number_value+prod_name+"long.csv"
    collection=db[name]
    documents = list(collection.find())

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(documents)
    print(df.head())

    to_drop = ['_id','ds','y']
    # # Check if columns to drop are present in dataframe
    df = df.drop(to_drop, axis=1)
    # Convert the DataFrame to a CSV file and return it as a Flask response
    csv_data = df.to_csv(index=False)
    # response = make_response(csv_data)
    # response.headers['Content-Disposition'] = 'attachment; filename=data.csv'
    # response.headers['Content-Type'] = 'text/csv'

    return jsonify({'file': csv_data})

@app.route('/pie-chart', methods=['POST'])
def pie_chart():
    data = request.get_json()
    filename = data['filename']
    collection = db[filename]
    df = pd.DataFrame(list(collection.find()))
    grouped = df.groupby(['PRODUCT_NAME'])['SALES'].sum()

    # create the new dataframe with the required columns
    new_df = pd.DataFrame({'PRODUCT_NAME': grouped.index, 'SALES': grouped.values})
    final_csv = new_df.to_csv(index=False)
    return jsonify({'file': final_csv})


app.debug=True

if __name__=='__main__':
    app.run(host='localhost',port=5000)