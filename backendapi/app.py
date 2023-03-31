import csv
import chardet
import numpy as np

import pymongo
from flask import Flask, request, jsonify, session
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
import pymongo
from pylab import rcParams
import statsmodels.api as sm
from dateutil import parser
from sklearn.metrics import mean_squared_error
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
    result = collection.insert_one(new_user)
    inserted_user = collection.find_one({'_id': ObjectId(result.inserted_id)})
    inserted_user['_id'] = str(inserted_user['_id'])
    return jsonify(inserted_user), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = collection.find_one({'email': data['email'], 'password': data['password']})
    if user:
        user['_id'] = str(user['_id'])
        return jsonify({'result': True,'user_id': str(user['_id'])})
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
            return jsonify({'response': False}), 400
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
        return "User record not found"
    print(column_value)
    print(time_period)
    print(number_value)
    # Detect the encoding of the file
    # file_data = file.read()
    # encoding = chardet.detect(file_data)['encoding']

    # Convert the file data to CSV format
    # file_data = file_data.decode(encoding)
    # csv_data = []
    # for row in csv.reader(file_data.splitlines()):
    #     csv_data.append(row)
    #
    # # Create a new collection under the user record
    # collection_name = title
    # collection = db[collection_name]
    #
    # # Insert the new record into the collection
    # new_record = {
    #     'title':title,
    #     'column_value': column_value,
    #     'time_period': time_period,
    #     'number_value': number_value,
    #     'file_data': csv_data
    # }
    # collection.insert_one(new_record)
    #
    # # Add the new collection to the user record
    # user_record = db.users.find_one({'_id': ObjectId(current_user)})
    # if 'datasets' not in user_record:
    #     user_record['datasets'] = {}
    # user_record['datasets'][collection_name] = collection_name
    # db.users.update_one({'_id': ObjectId(current_user)}, {'$set': {'datasets': user_record['datasets']}})
    # #ml model
    #

    ##AI MODEL
    # if (time_period == 'YEARLY'):
    #     freq = 'Y'
    # elif (time_period == 'MONTHLY'):
    #         freq = 'M'
    # elif (time_period == 'WEEKLY'):
    #         freq = 'W'
    # elif(time_period=="DAY_WISE"):
    #         freq = 'D'
    # data = pd.read_csv(file, encoding='latin-1')
    #
    # to_drop = ['ADDRESS_LINE2', 'STATE', 'POSTAL_CODE', 'TERRITORY', 'PRODUCT_CODE', 'CUSTOMER_NAME', 'PHONE',
    #            'ADDRESS_LINE1', 'CITY', 'CONTACT_LAST_NAME', 'CONTACT_FIRST_NAME']
    # # Check if columns to drop are present in dataframe
    # data = data.drop(to_drop, axis=1)
    # data['STATUS'].unique()
    # data['STATUS'] = pd.factorize(data.STATUS)[0] + 1
    # data['PRODUCT_LINE'].unique()
    # data['PRODUCT_LINE'] = pd.factorize(data.PRODUCT_LINE)[0] + 1
    # data['COUNTRY'].unique()
    # data['COUNTRY'] = pd.factorize(data.COUNTRY)[0] + 1
    # data['DEAL_SIZE'].unique()
    # data['DEAL_SIZE'] = pd.factorize(data.DEAL_SIZE)[0] + 1
    # data['ORDER_DATE'] = pd.to_datetime(data['ORDER_DATE'])
    # df = pd.DataFrame(data)
    # data.sort_values(by=['ORDER_DATE'], inplace=True)
    # data.set_index('ORDER_DATE', inplace=True)
    # df.sort_values(by=['ORDER_DATE'], inplace=True, ascending=True)
    # df.set_index('ORDER_DATE', inplace=True)
    # # df[column_value] = pd.to_numeric(df[column_value], errors='coerce')
    # new_data = pd.DataFrame(df[column_value])
    # new_data = pd.DataFrame(new_data[column_value].resample(freq).mean())
    # new_data = new_data.interpolate(method='linear')
    #
    #     # Method to Checking for Stationary: A stationary process has the property that the mean, variance and autocorrelation structure do not change over time.
    # train, test, validation = np.split(new_data[column_value].sample(frac=1),
    #                                        [int(.6 * len(new_data[column_value])), int(.8 * len(new_data[column_value]))])
    # print('Train Dataset')
    # print(train)
    # print('Test Dataset')
    # print(test)
    # print('Validation Dataset')
    # print(validation)
    #
    #     # SARIMA MODEL
    # mod = sm.tsa.statespace.SARIMAX(new_data,
    #                                     order=(1, 1, 1),
    #                                     seasonal_order=(1, 1, 1, 12),
    #                                     enforce_invertibility=False)
    # results = mod.fit()
    # pred = results.get_prediction()
    # if (freq == 'D'):
    #     pred = results.get_prediction(start=pd.to_datetime('2003-01-06'), dynamic=False)
    # pred.conf_int()
    # y_forecasted = pred.predicted_mean
    # y_truth = new_data[column_value]
    #
    # mse = mean_squared_error(y_truth, y_forecasted)
    # rmse = sqrt(mse)
    # mae = metrics.mean_absolute_error(y_forecasted, y_truth)
    # mape = metrics.mean_absolute_percentage_error(y_truth, y_forecasted)
    # mape = round(mape * 100, 2)
    # forecast = results.forecast(steps=int(number_value))
    # forecast = forecast.astype('float')
    # forecast_df = forecast.to_frame()
    # forecast_df.reset_index(level=0, inplace=True)
    # forecast_df.columns = ['PredictionDate', 'PredictedColumn']
    # print(forecast_df)
    # frame = pd.DataFrame(forecast_df)
    # frameDict = frame.to_dict('records')
    #
    # predicted_date = []
    # predicted_column = []
    # for i in range(0, len(frameDict)):
    #         predicted_column.append(frameDict[i]['PredictedColumn'])
    #         tempStr = str(frameDict[i]['PredictionDate'])
    #         dt = parser.parse(tempStr)
    #         predicted_date.append(
    #         dt.strftime('%A')[0:3] + ', ' + str(dt.day) + ' ' + dt.strftime("%b")[0:3] + ' ' + str(dt.year))
    #         # Find the user's record based on their email




    return jsonify({'response': True}), 200

@app.route('/retrivefiles',methods=['POST'])
def get_filenames():
    data = request.get_json()
    print(data['currentuser'])
    user = db.users.find_one({'_id': ObjectId(data['currentuser'])})
    if user is None:
        return jsonify({'error': 'User not found'}), 404
    files = user.get('datasets', {})
    filenames = [x for x in files.keys()]
    print(filenames)
    return jsonify({'filenames': filenames}), 200

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

    # print the collections
    data = pd.DataFrame(list(collection.find()))
    #AI MODEL
    if (time_period == 'YEARLY'):
        freq = 'Y'
    elif (time_period == 'MONTHLY'):
            freq = 'M'
    elif (time_period == 'WEEKLY'):
            freq = 'W'
    elif(time_period=="DAY_WISE"):
            freq = 'D'

    to_drop = ['ADDRESS_LINE2', 'STATE', 'POSTAL_CODE', 'TERRITORY', 'PRODUCT_CODE', 'CUSTOMER_NAME', 'PHONE',
               'ADDRESS_LINE1', 'CITY', 'CONTACT_LAST_NAME', 'CONTACT_FIRST_NAME']
    # Check if columns to drop are present in dataframe
    data = data.drop(to_drop, axis=1)
    data['STATUS'].unique()
    data['STATUS'] = pd.factorize(data.STATUS)[0] + 1
    data['PRODUCT_LINE'].unique()
    data['PRODUCT_LINE'] = pd.factorize(data.PRODUCT_LINE)[0] + 1
    data['COUNTRY'].unique()
    data['COUNTRY'] = pd.factorize(data.COUNTRY)[0] + 1
    data['DEAL_SIZE'].unique()
    data['DEAL_SIZE'] = pd.factorize(data.DEAL_SIZE)[0] + 1
    data['ORDER_DATE'] = pd.to_datetime(data['ORDER_DATE'])
    df = pd.DataFrame(data)
    data.sort_values(by=['ORDER_DATE'], inplace=True)
    data.set_index('ORDER_DATE', inplace=True)
    df.sort_values(by=['ORDER_DATE'], inplace=True, ascending=True)
    df.set_index('ORDER_DATE', inplace=True)
    # df[column_value] = pd.to_numeric(df[column_value], errors='coerce')
    new_data = pd.DataFrame(df[column_value])
    new_data = pd.DataFrame(new_data[column_value].resample(freq).mean())
    new_data = new_data.interpolate(method='linear')

        # Method to Checking for Stationary: A stationary process has the property that the mean, variance and autocorrelation structure do not change over time.
    train, test, validation = np.split(new_data[column_value].sample(frac=1),
                                           [int(.6 * len(new_data[column_value])), int(.8 * len(new_data[column_value]))])
    print('Train Dataset')
    print(train)
    print('Test Dataset')
    print(test)
    print('Validation Dataset')
    print(validation)

        # SARIMA MODEL
    mod = sm.tsa.statespace.SARIMAX(new_data,
                                        order=(1, 1, 1),
                                        seasonal_order=(1, 1, 1, 12),
                                        enforce_invertibility=False)
    results = mod.fit()
    pred = results.get_prediction()
    if (freq == 'D'):
        pred = results.get_prediction(start=pd.to_datetime('2003-01-06'), dynamic=False)
    pred.conf_int()
    y_forecasted = pred.predicted_mean
    y_truth = new_data[column_value]

    mse = mean_squared_error(y_truth, y_forecasted)
    rmse = sqrt(mse)

    mae = metrics.mean_absolute_error(y_forecasted, y_truth)
    mape = metrics.mean_absolute_percentage_error(y_truth, y_forecasted)
    mape = round(mape * 100, 2)
    forecast = results.forecast(steps=int(number_value))
    forecast = forecast.astype('float')
    forecast_df = forecast.to_frame()
    forecast_df.reset_index(level=0, inplace=True)
    forecast_df.columns = ['PredictionDate', 'PredictedColumn']
    print(forecast_df)
    frame = pd.DataFrame(forecast_df)
    frameDict = frame.to_dict('records')

    predicted_date = []
    predicted_column = []
    for i in range(0, len(frameDict)):
            predicted_column.append(frameDict[i]['PredictedColumn'])
            tempStr = str(frameDict[i]['PredictionDate'])
            dt = parser.parse(tempStr)
            predicted_date.append(
            dt.strftime('%A')[0:3] + ', ' + str(dt.day) + ' ' + dt.strftime("%b")[0:3] + ' ' + str(dt.year))
            # Find the user's record based on their email
    print("mae",mae)
    print("mse",mse)
    print("rmse",rmse)
    print("mape",mape)
    print(np.mean(new_data))

    return jsonify({'response': True}), 200

app.debug=True

if __name__=='__main__':
    app.run(host='localhost',port=5000)