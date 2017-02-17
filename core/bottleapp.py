from bottle import route, run, Bottle, error
from bottle.ext.mongo import MongoPlugin
from bson.json_util import dumps

app = Bottle()
mongo = MongoPlugin(uri='mongodb://localhost:27017', db='restdb', json_mongo=True)
app.install(mongo)


### RAWSHOTS ###
#Stores raw copies of shots directly from Android device
@app.route('/rawshots', methods=['GET'])
def getAllShots(mongodb):
    shots = mongodb['rawshots']
    output = []
    for s in shots.find():
        output.append(dumps(s))
    return dumps({'rawshots': output})

@app.route('/rawshot/<shot_id:int>', methods=['GET'])
def getShot(shot_id, mongodb):
    shot = mongodb['rawshots'].find({'_id':shot_id})
    return dumps({'RawShot': shot})

@app.route('/rawshot/<shot_id:int>', methods=['DELETE'])
def deleteShot(shot_id, mongodb):
    mongodb['rawshots'].remove({'_id':shot_id}, 1)
    return dumps({'Result': True})

@app.route('/rawshots', methods=['POST'])
def createShot(mongodb):
    if not request.json:
        abort(400)
    shots = mongodb['rawshots']
    userID = request.json['userID']
    upperAccel = request.json['upperAccel']
    upperGyro = request.json['upperGyro']
    lowerAccel = request.json['lowerAccel']
    lowerGyro = request.json['lowerGyro']
    new_shot_id = shots.insert({'userID':userID, 'upperGyro':upperGyro, 'upperAccel':upperAccel, 'lowerGyro':lowerGyro, 'lowerAccel':lowerAccel}).inserted_id
    new_shot = shots.find({'_id':new_shot_id})

    return dumps({'shotID': new_shot_id}), 201

### FUSED SHOTS ###
@app.route('/fusedshots', methods=['GET'])
def getAllShots(mongodb):
    shots = mongodb['fusedshots']
    output = []
    for s in shots.find():
        output.append(dumps(s))
    return dumps({'rawshots': output})

@app.route('/fusedshot/<shot_id:int>', methods=['GET'])
def getShot(shot_id, mongodb):
    shot = mongodb['fusedshots'].find({'_id':shot_id})
    return dumps({'RawShot': shot})

@app.route('/fusedshot/<shot_id:int>', methods=['DELETE'])
def deleteShot(shot_id, mongodb):
    mongodb['fusedshots'].remove({'_id':shot_id}, 1)
    return dumps({'Result': True})

### ML_DATA ###
@app.route('/mlresults', methods=['GET'])
def getAllShots(mongodb):
    results = mongodb['mlresults']
    output = []
    for s in results.find():
        output.append(dumps(s))
    return dumps({'ML Results': output})

@app.route('/mlresult/<shot_id:int>', methods=['GET'])
def getShot(shot_id, mongodb):
    shot = mongodb['mlresults'].find({'_id':shot_id})
    return dumps({'ML Result': shot})

@app.route('/mlresult/<shot_id:int>', methods=['DELETE'])
def deleteShot(shot_id, mongodb):
    mongodb['mlresults'].remove({'_id':shot_id}, 1)
    return dumps({'Result': True})


### USERS ###
#Database of all Users
@app.route('/user/<user_id:int>', methods=['GET'])
def getUser(user_id, mongodb):
    user = mongodb['users'].find({'_id':user_id})
    return dumps({'User': user})

@app.route('/user/<user_id:int>', methods=['DELETE'])
def deleteUser(user_id, mongodb):
    mongodb['users'].remove({'_id':user_id}, 1)
    return dumps({'Result': True})

@app.route('/users', methods=['POST'])
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
    return "The Silkiest Mitties - APIS: /users, /rawshots, /fusedshots, /mlresult"

@error(404)
def not_found(error):
    return dumps({'404': 'Resource Not Found'})

if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
