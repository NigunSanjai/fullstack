from flask import Flask,request
from flask_restful import Api, Resource,reqparse
from models import UserModel,db
from flask_cors import CORS,cross_origin
app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

api = Api(app)
db.init_app(app)

@app.before_first_request
def create_table():
    db.create_all()

class UserView(Resource):
    # parser = reqparse.RequestParser()
    # parser.add_argument('name',type=str,required=True,help="Required Field")
    # parser.add_argument('email', type=str, required=True, help="Required Field")
    # parser.add_argument('password', type=str, required=True, help="Required Field")
    # parser.add_argument('contact', type=int, required=True, help="Required Field")
    def get(self):
        users = UserModel.query.all()
        return{'Users': list(x.json() for x in users)}
    def post(self):
        data = request.get_json()
        new_user = UserModel(data['name'],data['email'],data['password'],data['contact'])
        db.session.add(new_user)
        db.session.commit()
        db.session.flush()
        return new_user.json(),201


class SingleUserView(Resource):

    def get(self,id):
        user = UserModel.query.filter_by(id=id).first()
        if(user):
            return user.json()
        return{'message':"User not found"},404

    def delete(self,id):
        user = UserModel.query.filter_by(id=id).first()
        if user:
            db.session.delete(user)
            db.session.commit()
            return {'message':'Deleted'}
        else:
            return {'message': "User not found"}, 404
    def put(self,id):
        data = request.get_json()
        user = UserModel.query.filter_by(id=id).first()
        if user:
            user.name = data['name']
            user.email = data['email']
            user.password = data['password']
            user.contact = data['contact']
        else:
            user = UserModel(id=id,**data)
        db.session.add(user)
        db.session.commit()

        return user.json()


api.add_resource(UserView,'/users')
api.add_resource(SingleUserView ,'/user/<int:id>')
api.add_resource(SingleUserView,'/checkusers')

app.debug=True

if __name__=='__main__':
    app.run(host='localhost',port=5000)