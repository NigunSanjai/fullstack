import csv

import pymongo
from flask import Flask, request, jsonify, session
from flask_restful import Api, Resource
from models import UserModel, db
from flask_cors import CORS, cross_origin
from flask_pymongo import PyMongo
from pymongo import MongoClient
from bson.json_util import dumps
from bson.objectid import ObjectId

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
    current_user = request.form.get('currentuser')
    column_value = request.form.get('column_value')
    time_period = request.form.get('time_period')
    number_value = request.form.get('number_value')
    file = request.files['file']

    # Convert the file data to CSV format
    file_data = file.read().decode('utf-8').splitlines()
    csv_reader = csv.reader(file_data)
    csv_data = []
    for row in csv_reader:
        csv_data.append(row)
    db.users.update_one(
    {"_id": ObjectId(current_user)},
    {"$set": {
            'current_user': current_user,
            'column_value': column_value,
            'time_period': time_period,
            'number_value': number_value,
            'file_data': csv_data
        }}
        )
    return jsonify({'response': True}), 200
app.debug=True

if __name__=='__main__':
    app.run(host='localhost',port=5000)