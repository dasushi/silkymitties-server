from bottle import route, run, Bottle, error
from bottle.ext.mongo import MongoPlugin
from bson.json_util import dumps

app = Bottle()
mongo = MongoPlugin(uri='mongodb://localhost:27017', db='restdb', json_mongo=True)
app.install(mongo)


SENSORTYPE = (
    ('a', 'ACCEL'),
    ('g', 'GYRO'),
    ('f', 'FUSION'),
)

HANDEDNESS = (
    ('l', 'LEFT'),
    ('r', 'RIGHT'),
)


### RAWSHOTS ###
#Stores raw copies of shots directly from Android device
@app.route('/silkymitties/rawshots', methods=['GET'])
def getAllShots(mongodb):
    shots = mongodb['shots']
    output = []
    for s in shots.find():
        output.append(dumps(s))
    return dumps({'Shots': output})

@app.route('/silkymitties/rawshot/<shot_id:int>', methods=['GET'])
def getShot(shot_id, mongodb):
    shot = mongodb['shots'].find({'_id':shot_id})
    return dumps({'Shot': shot})

@app.route('/silkymitties/shot/<shot_id:int>', methods=['DELETE'])
def deleteShot(shot_id, mongodb):
    mongodb['shots'].remove({'_id':shot_id}, 1)
    return dumps({'Result': True})

@app.route('/silkymitties/rawshots', methods=['POST'])
def createShot(mongodb):
    if not request.json:
        abort(400)
    shots = mongodb['shots']
    userID = request.json['userID']
    upperAccel = request.json['upperAccel']
    upperGyro = request.json['upperGyro']
    lowerAccel = request.json['lowerAccel']
    lowerGyro = request.json['lowerGyro']
    new_shot_id = shots.insert({'userID':userID, 'upperGyro':upperGyro, 'upperAccel':upperAccel, 'lowerGyro':lowerGyro, 'lowerAccel':lowerAccel}).inserted_id
    new_shot = shots.find({'_id':new_shot_id})

    return dumps({'shotID': new_shot_id, 'userID':userID}), 201

### FUSED SHOTS ###

### ML_DATA ###



### USERS ###
#Database of all Users

@app.route('/silkymitties/user/<user_id:int>', methods=['GET'])
def getUser(user_id, mongodb):
    user = mongodb['users'].find({'_id':user_id})
    return dumps({'User': user})

@app.route('/silkymitties/user/<user_id:int>', methods=['DELETE'])
def deleteUser(user_id, mongodb):
    mongodb['users'].remove({'_id':user_id}, 1)
    return dumps({'Result': True})

@app.route('/silkymitties/users', methods=['POST'])
def createUser(mongodb):
    if not request.json:
        abort(400)
    users = mongodb['users']
    firstName = request.json['firstName']
    lastName = request.json['lastName']
    age = request.json['age']
    handedness = request.json['handedness']
    height = request.json['height']
    new_user_id = users.insert({'firstName':firstName, 'lastName':lastName, 'age':age, 'handedness':handedness, 'height':height}).inserted_id
    new_user = users.find({'_id':new_user_id})
    return dumps({'userID': new_user_id}), 201

@app.route('/')
def index():
    return "The Silkiest Mitties"

@error(404)
def not_found(error):
    return dumps({'404': 'Resource Not Found'})

if __name__=='__main__':
    app.run(debug=True)
