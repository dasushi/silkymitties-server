#!flask/bin/python
import flask
import pandas
from django.db import models

app = flask.Flask(__name__)

class sensorEntry(models.Model):
    shotID = models.IntegerField()
    timestamp = models.IntegerField()
    xVal = models.FloatField()
    yVal = models.FloatField()
    zVal = models.FloatField()

class rawShotEntry(models.Model):
    shotID = models.IntegerField()
    userID = models.CharField(max_length = 30)

    'upperHand':{
        'accVals':{'time':{}, 'x-acc':{}, 'y-acc':{}, 'z-acc':{}},
        'gyroVals':{'time':{}, 'x-gyro':{}, 'y-gyro':{}, 'z-gyro':{}},
        'deviceID':u'rg132132'
    },
    'lowerHand':{
        'accVals':{'time':{}, 'x-acc':{}, 'y-acc':{}, 'z-acc':{}},
        'gyroVals':{'time':{}, 'x-gyro':{}, 'y-gyro':{}, 'z-gyro':{}},
        'deviceID':u'rg132132'

shots = [
    {
        'shotID':1,
        'userID':u'testuser',
        'upperHand':{
            'accVals':{'time':{}, 'x-acc':{}, 'y-acc':{}, 'z-acc':{}},
            'gyroVals':{'time':{}, 'x-gyro':{}, 'y-gyro':{}, 'z-gyro':{}},
            'deviceID':u'rg132132'
        },
        'lowerHand':{
            'accVals':{'time':{}, 'x-acc':{}, 'y-acc':{}, 'z-acc':{}},
            'gyroVals':{'time':{}, 'x-gyro':{}, 'y-gyro':{}, 'z-gyro':{}},
            'deviceID':u'rg132132'
        }
    },
    {
        'shotID':2,
        'userID':u'testuser',
        'upperHand':{
            'accVals':{'time':{}, 'x-acc':{}, 'y-acc':{}, 'z-acc':{}},
            'gyroVals':{'time':{}, 'x-gyro':{}, 'y-gyro':{}, 'z-gyro':{}},
            'deviceID':u'rg132132'
        },
        'lowerHand':{
            'accVals':{'time':{}, 'x-acc':{}, 'y-acc':{}, 'z-acc':{}},
            'gyroVals':{'time':{}, 'x-gyro':{}, 'y-gyro':{}, 'z-gyro':{}},
            'deviceID':u'rg132132'
        }
    }

]

@app.route('/silkymitties/api/v0.1/shots', methods=['GET'])
def getShots():
    return flask.jsonify({'Shots': shots})

@app.route('/silkymitties/api/v0.1/shot/<int:shot_id>', methods=['GET'])
def getShot(shot_id):
    shot = [shot for shot in shots if shots['shotID'] == shot_id]
    if len(shot)==0:
        flask.abort(404)
    return flask.jsonify({'Shot': shot})

@app.route('/silkymitties/api/v0.1/shot/<int:shot_id>', methods=['DELETE'])
def deleteShot(shot_id):
    shot = [shot for shot in shots if shots['shotID'] == shot_id]
    if len(shot)==0:
        flask.abort(404)
    shots.remove(shot[0])
    return flask.jsonify({'Result': True})

@app.route('/silkymitties/api/v0.1/shot/<int:shot_id>', methods=['PUT'])
def updateShot(shot_id):
    shot = [shot for shot in shots if shots['shotID'] == shot_id]
    if len(shot)==0:
        flask.abort(404)
    if not request.json:
        flask.abort(400)
    if 'userID' in request.json and type(request.json['userID']) is not int:
        flask.abort(400)
    shot[0]['userID'] = request.json['userID']
    shot[0]['upperHand'] = request.json['upperHand']
    shot[0]['lowerHand'] = request.json['lowerHand']
    return flask.jsonify({'shot': shot[0]})

@app.route('/silkymitties/api/v0.1/shots', methods=['POST'])
def createShot():
    if not request.json or not 'userID' in request.json:
        abort(400)
    shot = {
        'shotID':shots[-1]['shotID'] + 1,
        'userID':request.json['userID'],
        'upperHand':request.json['upperHand'],
        'lowerHand':request.json['lowerHand']
    }
    shots.append(shot)
    return flask.jsonify({'Shot': shot}), 201

@app.route('/')
def index():
    return "The Silkiest Mitties"

@app.errorhandler(404)
def not_found(error):
    return flask.make_response(flask.jsonify({'error': 'Resource Not Found'}), 404)

if __name__=='__main__':
    app.run(debug=True)
