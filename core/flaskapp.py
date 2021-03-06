#!flask/bin/python
from flask import Flask, jsonify, make_response
from flask_pymongo import PyMongo
#import pandas
#from django.db import models
#from djangotoolbox.fields import ListField, EmbeddedModelField

app = Flask(__name__)

app.config['MONGO_DBNAME'] = 'restdb'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/restdb'

mongo = PyMongo(app)

SENSORTYPE = (
    ('a', 'ACCEL'),
    ('g', 'GYRO'),
    ('f', 'FUSION'),
)

HANDEDNESS = (
    ('l', 'LEFT'),
    ('r', 'RIGHT'),
)

#class fusionLog(models.Model):
#    shotID = models.ForeignKey(fusionShotLog, on_delete = models.CASCADE)
#    xVal = ListField()
#    yVal = ListField()
#    zVal = ListField()

#class sensorLog(models.Model):
#    shotID = models.ForeignKey(shotLog, on_delete = models.CASCADE)
#    timestamp = ListField()
#    xVal = ListField()
#    yVal = ListField()
#    zVal = ListField()

#class fusionLog(models.Model):
#    userID = models.ForeignKey(userProfile, on_delete = models.SET_NULL)
#    shotDate = models.DateTimeField(auto_now_add=True, null=True)
#    upperAccel = models.ForeignKey(fusionLog, on_delete = models.CASCADE)
#    upperGyro = models.ForeignKey(fusionLog, on_delete = models.CASCADE)
#    lowerAccel = models.ForeignKey(fusionLog, on_delete = models.CASCADE)
#    lowerGyro = models.ForeignKey(fusionLog, on_delete = models.CASCADE)


#class shotLog(models.Model):
#    userID = models.ForeignKey(userProfile, on_delete = models.SET_NULL)
#    shotDate = models.DateTimeField(auto_now_add=True, null=True)
#    upperAccel = models.ForeignKey(sensorLog, on_delete = models.CASCADE)
#    upperGyro = models.ForeignKey(sensorLog, on_delete = models.CASCADE)
#    lowerAccel = models.ForeignKey(sensorLog, on_delete = models.CASCADE)
#    lowerGyro = models.ForeignKey(sensorLog, on_delete = models.CASCADE)

#class userProfile(models.Model):
#    firstName = models.CharField(max_length=30)
#    lastName = models.CharField(max_length=30)
#    age = models.IntegerField()
#    height = models.IntegerField()
#    handedness = models.CharField(max_length=1, choices=HANDEDNESS)


### RAWSHOTS ###
#Stores raw copies of shots directly from Android device
@app.route('/silkymitties/rawshots', methods=['GET'])
def getAllShots():
    shots = mongo.db.shots
    output = []
    for s in shots.find():
        output.append(Flask.jsonify(s))
    return jsonify({'Shots': output})

@app.route('/silkymitties/rawshot/<int:shot_id>', methods=['GET'])
def getShot(shot_id):
    shot = mongo.db.shots.find_one_or_404({'_id':shot_id})
    return jsonify({'Shot': shot})

@app.route('/silkymitties/shot/<int:shot_id>', methods=['DELETE'])
def deleteShot(shot_id):
    mongo.db.shots.remove({'_id':shot_id}, 1)
    return jsonify({'Result': True})

@app.route('/silkymitties/rawshots', methods=['POST'])
def createShot():
    if not request.json:
        abort(400)
    shots = mongo.db.shots
    userID = request.json['userID']
    upperAccel = request.json['upperAccel']
    upperGyro = request.json['upperGyro']
    lowerAccel = request.json['lowerAccel']
    lowerGyro = request.json['lowerGyro']
    new_shot_id = shots.insert_one({'userID':userID, 'upperGyro':upperGyro, 'upperAccel':upperAccel, 'lowerGyro':lowerGyro, 'lowerAccel':lowerAccel}).inserted_id
    new_shot = shots.find_one({'_id':new_shot_id})

    return jsonify({'shotID': new_shot_id, 'userID':userID}), 201

### FUSED SHOTS ###

### ML_DATA ###



### USERS ###
#Database of all Users

@app.route('/silkymitties/user/<int:user_id>', methods=['GET'])
def getUser(user_id):
    user = mongo.db.users.find_one_or_404({'_id':user_id})
    return jsonify({'User': user})

@app.route('/silkymitties/user/<int:user_id>', methods=['DELETE'])
def deleteUser(user_id):
    mongo.db.users.remove({'_id':user_id}, 1)
    return jsonify({'Result': True})

@app.route('/silkymitties/users', methods=['POST'])
def createUser():
    if not request.json:
        abort(400)
    users = mongo.db.users
    firstName = request.json['firstName']
    lastName = request.json['lastName']
    age = request.json['age']
    handedness = request.json['handedness']
    height = request.json['height']
    new_user_id = users.insert_one({'firstName':firstName, 'lastName':lastName, 'age':age, 'handedness':handedness, 'height':height}).inserted_id
    new_user = users.find_one({'_id':new_user_id})
    return jsonify({'userID': new_user_id}), 201

@app.route('/')
def index():
    return "The Silkiest Mitties"

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Resource Not Found'}), 404)

if __name__=='__main__':
    app.run(debug=True)
