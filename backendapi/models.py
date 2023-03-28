
from flask_sqlalchemy import SQLAlchemy

db=SQLAlchemy()

class UserModel(db.Model):
    __tablename__="users"
    id=db.Column(db.Integer,primary_key=True)
    name=db.Column(db.String(80))
    email=db.Column(db.String(80))
    password=db.Column(db.String(80))
    contact=db.Column(db.Integer())

    def __init__(self, name, email, password, contact):
        self.name=name
        self.email = email
        self.password = password
        self.contact = contact


    def json(self):
        return {"name":self.name,"email":self.email,"password":self.password,"contact":self.contact}
