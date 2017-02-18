from bottle import route, run, Bottle, error, request
from bottle.ext.mongo import MongoPlugin
from bson.json_util import dumps
from bson.objectid import ObjectId
from fusion import processRawShot, processLabelledShot
#from lstm import predictResult

app = Bottle()
mongo = MongoPlugin(uri='mongodb://127.0.0.1', db='restdb', json_mongo=True)
app.install(mongo)

#API:
#/rawshots, /rawshots/user/<id>, /rawshot/<id>
#/fusedshots, /fusedshots/user/<id>, /fusedshot/<id>
#/mlresults, /mlresult/user/<id>, /mlresults/<id>
#/users, /user/<id>



### RAWSHOTS ###
#Stores raw copies of shots directly from Android device WITH labelled data
#for training purposes, does not process or return an ML result
@app.route('/rawshots', method='GET')
def getAllShots(mongodb):
    shots = mongodb['rawshots']
    output = []
    for s in shots.find():
        output.append(dumps(s))
    return dumps(output)

@app.route('/rawshots/user/<user_id>', method='GET')
def getAllShots(user_id, mongodb):
    shots = mongodb['rawshots']
    output = []
    for s in shots.find({'userID':user_id}):
        output.append(dumps(s))
    return dumps(output)

@app.route('/rawshot/<shot_id>', method='GET')
def getShot(shot_id, mongodb):
    shot = mongodb['rawshots'].find_one({'_id':ObjectId(shot_id)})
    return dumps(shot)

@app.route('/rawshot/<shot_id>', method='DELETE')
def deleteShot(shot_id, mongodb):
    mongodb['rawshots'].remove({'_id':ObjectId(shot_id)}, 1)
    return {'Result': 'True'}

@app.route('/rawshots', method='PUT')
def createShot(mongodb):
    if not request.json:
        abort(400)
    shots = mongodb['rawshots']
    userID = request.json['userID']
    upperAccel = request.json['upperAccel']
    upperGyro = request.json['upperGyro']
    lowerAccel = request.json['lowerAccel']
    lowerGyro = request.json['lowerGyro']
    new_shot_id = shots.insert_one({'userID':userID, 'upperGyro':upperGyro, 'upperAccel':upperAccel, 'lowerGyro':lowerGyro, 'lowerAccel':lowerAccel}).inserted_id
    new_shot = shots.find({'_id':new_shot_id})

    mlResultID = processRawShot(new_shot_id)

    return dumps({'rawshotID': new_shot_id, 'fusedshotID': fused_id})

### LABELLEDSHOTS ###
#Stores raw copies of shots directly from Android device WITH labelled data
#for training purposes, does not process or return an ML result
@app.route('/labelledshots', method='GET')
def getAllShots(mongodb):
    shots = mongodb['labelledshots']
    output = []
    for s in shots.find():
        output.append(dumps(s))
    return dumps(output)

@app.route('/labelledshots/user/<user_id>', method='GET')
def getAllShots(user_id, mongodb):
    shots = mongodb['labelledshots']
    output = []
    for s in shots.find({'userID':user_id}):
        output.append(dumps(s))
    return dumps(output)

@app.route('/labelledshot/<shot_id>', method='GET')
def getShot(shot_id, mongodb):
    shot = mongodb['labelledshots'].find_one({'_id':ObjectId(shot_id)})
    return dumps(shot)

@app.route('/labelledshot/<shot_id>', method='DELETE')
def deleteShot(shot_id, mongodb):
    mongodb['labelledshots'].remove({'_id':ObjectId(shot_id)}, 1)
    return {'Result': 'True'}

@app.route('/labelledshots', method='PUT')
def createShot(mongodb):
    if not request.json:
        abort(400)
    shots = mongodb['labelledshots']
    userID = request.json['userID']
    upperAccel = request.json['upperAccel']
    upperGyro = request.json['upperGyro']
    lowerAccel = request.json['lowerAccel']
    lowerGyro = request.json['lowerGyro']
    shotType = request.json['shotType']
    new_shot_id = shots.insert_one({'userID':userID, 'upperGyro':upperGyro, 'upperAccel':upperAccel, 'lowerGyro':lowerGyro, 'lowerAccel':lowerAccel, 'shotType': shotType}).inserted_id
    new_shot = shots.find({'_id':new_shot_id})

    fused_id = processLabelledShot(new_shot_id)

    return dumps({'labelledshotID': new_shot_id, 'lblfusedshotID': fused_id})

### FUSED SHOTS ###
@app.route('/fusedshots', method='GET')
def getAllShots(mongodb):
    shots = mongodb['fusedshots']
    output = []
    for s in shots.find():
        output.append(dumps(s))
    return dumps(output)

@app.route('/fusedshots/user/<user_id>', method='GET')
def getAllShots(user_id, mongodb):
    shots = mongodb['fusedshots']
    output = []
    for s in shots.find({'_id':ObjectId(user_id)}):
        output.append(dumps(s))
    return dumps(output)

@app.route('/fusedshot/<shot_id>', method='GET')
def getShot(shot_id, mongodb):
    shot = mongodb['fusedshots'].find_one({'_id':ObjectId(shot_id)})
    return dumps(shot)

@app.route('/fusedshot/<shot_id>', method='DELETE')
def deleteShot(shot_id, mongodb):
    mongodb['fusedshots'].remove({'_id':ObjectId(shot_id)}, 1)
    return {'Result': 'True'}

### LABELLED FUSED SHOTS ###
@app.route('/lblfusedshots', method='GET')
def getAllShots(mongodb):
    shots = mongodb['lblfusedshots']
    output = []
    for s in shots.find():
        output.append(dumps(s))
    return dumps(output)

@app.route('/lblfusedshots/user/<user_id>', method='GET')
def getAllShots(user_id, mongodb):
    shots = mongodb['lblfusedshots']
    output = []
    for s in shots.find({'userID':user_id}):
        output.append(dumps(s))
    return dumps(output)

@app.route('/lblfusedshot/<shot_id>', method='GET')
def getShot(shot_id, mongodb):
    shot = mongodb['lblfusedshots'].find_one({'_id':ObjectId(shot_id)})
    return dumps(shot)

@app.route('/lblfusedshot/<shot_id>', method='DELETE')
def deleteShot(shot_id, mongodb):
    mongodb['fusedshots'].remove({'_id':ObjectId(shot_id)}, 1)
    return {'Result': 'True'}

### ML_DATA ###
@app.route('/mlresults', method='GET')
def getAllShots(mongodb):
    results = mongodb['mlresults']
    output = []
    for s in results.find():
        output.append(dumps(s))
    return dumps(output)

@app.route('/mlresults/user/<user_id>', method='GET')
def getAllShots(user_id, mongodb):
    shots = mongodb['mlresults']
    output = []
    for s in shots.find({'_id':ObjectId(user_id)}):
        output.append(dumps(s))
    return dumps(output)

@app.route('/mlresult/<shot_id>', method='GET')
def getShot(shot_id, mongodb):
    shot = mongodb['mlresults'].find_one({'_id':ObjectId(shot_id)})
    return dumps(shot)

@app.route('/mlresult/<shot_id>', method='DELETE')
def deleteShot(shot_id, mongodb):
    mongodb['mlresults'].remove({'_id':ObjectId(shot_id)}, 1)
    return {'Result': 'True'}


### USERS ###
#Database of all Users
@app.route('/users', method='GET')
def getAllUsers(mongodb):
    results = mongodb['users']
    output = []
    for s in results.find():
        output.append(dumps(s))
    return dumps(output)

@app.route('/users', method='DELETE')
def getAllUsers(mongodb):
    results = mongodb['users']
    output = []
    for s in results.find():
        output.append(dumps(s))
    return dumps(output)

@app.route('/user/<user_id>', method='GET')
def getUser(user_id, mongodb):
    user = mongodb['users'].find_one({'_id':ObjectId(user_id)})
    return dumps(user)

@app.route('/user/<user_id>', method='DELETE')
def deleteUser(user_id, mongodb):
    mongodb['users'].remove({'_id':user_id}, 1)
    return {'Result': 'True'}

@app.route('/users', method='PUT')
def createUser(mongodb):
    if not request.json:
        abort(400)
    users = mongodb['users']
    firstName = request.json['firstName']
    lastName = request.json['lastName']
    age = request.json['age']
    handedness = request.json['handedness']
    height = request.json['height']
    new_user_id = users.insert_one({'firstName':firstName, 'lastName':lastName, 'age':age, 'handedness':handedness, 'height':height}).inserted_id
    new_user = users.find({'_id':new_user_id})
    return dumps(new_user)

@app.route('/')
def index():
    return "The Silkiest Mitties - APIS: /users, /rawshots, /labelledshots, /fusedshots, /lblfusedshots, /mlresult"

@error(404)
def not_found(error):
    return dumps({'404': 'Resource Not Found'})

if __name__=='__main__':
    #private network run
    #app.run(debug=True, host='127.0.0.1', port=8080)
    #public network run
    app.run(debug=True, host='0.0.0.0', port=8080)
