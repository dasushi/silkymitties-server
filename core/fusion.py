import numpy as np
import pandas as pd
import math
import csv
import os
from timestamps import regularizeTimestamps
import matplotlib.pyplot as plt
from pymongo import MongoClient
import json

#input: [accel[timestamp, ax, ay, az], gyro [timestamp, gx, gy, gz]]
#output: interval, theta[tx, ty, tz]
#start_time = math.max(accel[0]['timestamp'], gyro[0]['timestamp'])
#end_time = math.min(accel[-1]['timestamp'], gyro[-1]['timestamp'])
#theta.length = min(start_time, end_time)

def loadJSONFolder():
    client = MongoClient('localhost', 27017)
    db = client['restdb']
    files = [f for f in os.listdir("json") if os.path.isfile(os.path.join("json", f))]
    total_data = []
    for f in files:
        with open(os.path.join("json", f)) as raw_data:
            jsondata = json.load(raw_data)
        print(f)
        shot = jsondata['shot']
        upperGyro = shot['upperGyro']
        upperAccel = shot['upperAccel']
        lowerAccel = shot['lowerAccel']
        lowerGyro = shot['lowerGyro']
        shotType = shot['type']
        print(shotType)
        handedness = shot['shoots']
        new_shot_id = db['testlabelledshots'].insert_one({'upperGyro':upperGyro, \
            'upperAccel':upperAccel, 'lowerGyro':lowerGyro, 'lowerAccel':lowerAccel, \
            'shotType': shotType, 'handedness':handedness}).inserted_id
        fused_id = processLabelledShot(new_shot_id)

        total_data.append(fused_id)


def loadFromFile(plot):
    with open("data/Board2_ACC_2.csv") as f:
        reader = csv.reader(f)
        orig_acc_data = np.array(list(reader))
        orig_acc_data = np.delete(orig_acc_data, [0,1], axis=0)
    with open("data/Board2_GYRO_2.csv") as f:
        reader = csv.reader(f)
        orig_gyr_data = list(reader)
        orig_gyr_data = np.delete(orig_gyr_data, [0,1], axis=0)
    with open("data/Board1_ACC_2.csv") as f:
        reader = csv.reader(f)
        orig_acc_data_2 = np.array(list(reader))
        orig_acc_data_2 = np.delete(orig_acc_data_2, [0,1], axis=0)
    with open("data/Board1_GYRO_2.csv") as f:
        reader = csv.reader(f)
        orig_gyr_data_2 = list(reader)
        orig_gyr_data_2 = np.delete(orig_gyr_data_2, [0,1], axis=0)
    upper_timestamps, upper_pair = processPair(orig_acc_data, orig_gyr_data, plot)
    lower_timestamps, lower_pair = processPair(orig_acc_data_2, orig_gyr_data_2, plot)
    printSummary(upper_timestamps, upper_pair, lower_timestamps, lower_pair)

def processRawShot(rawShotID):
    client = MongoClient('localhost', 27017)
    db = client['restdb']
    target = db['rawshots'].find_one(rawShotID)
    data = json.loads(target)
    upperAccel = np.column_stack((data['upperAccel']['timestamps'], data['upperAccel']['x'], data['upperAccel']['y'], data['upperAccel']['z']))
    lowerAccel = np.column_stack((data['lowerAccel']['timestamps'], data['lowerAccel']['x'], data['lowerAccel']['y'], data['lowerAccel']['z']))
    upperGyro = np.column_stack((data['upperGyro']['timestamps'], data['upperGyro']['x'], data['upperGyro']['y'], data['upperGyro']['z']))
    lowerGyro = np.column_stack((data['lowerGyro']['timestamps'], data['lowerGyro']['x'], data['lowerGyro']['y'], data['lowerGyro']['z']))
    upper_timestamps, upper_theta = processPair(upperAccel, upperGyro, False)
    lower_timestamps, lower_theta = processPair(lowerAccel, lowerGyro, False)

    uthetaX = upper_theta[:,0]
    uthetaY = upper_theta[:,1]
    uthetaZ = upper_theta[:,2]

    lthetaX = lower_theta[:,0]
    lthetaY = lower_theta[:,1]
    lthetaZ = lower_theta[:,2]

    fusedShotID = db['fusedshots'].insert_one({'upperTimestamps':upper_timestamps, \
    'lowerTimestamps':lower_timestamps, 'upperTheta':{'uthetaX':uthetaX, 'uthetaY':uthetaY, 'uthetaZ':uthetaZ}, \
    'lowerTheta':{'lthetaX':lthetaX, 'lthetaY':lthetaY, 'lthetaZ':lthetaZ},}).inserted_id

    return fusedShotID

def fuseLabelledShots():
    client = MongoClient('localhost', 27017)
    db = client['restdb']
    for data in db['labelledshots'].find():
        processLabelledShot(data['_id'])

def processLabelledShot(rawShotID):
    client = MongoClient('localhost', 27017)
    db = client['restdb']
    data = db['testlabelledshots'].find_one(rawShotID)
    shotType = data['shotType']
    upperAccel = np.column_stack((data['upperAccel']['timestamps'], data['upperAccel']['x'], data['upperAccel']['y'], data['upperAccel']['z']))
    lowerAccel = np.column_stack((data['lowerAccel']['timestamps'], data['lowerAccel']['x'], data['lowerAccel']['y'], data['lowerAccel']['z']))
    upperGyro = np.column_stack((data['upperGyro']['timestamps'], data['upperGyro']['x'], data['upperGyro']['y'], data['upperGyro']['z']))
    lowerGyro = np.column_stack((data['lowerGyro']['timestamps'], data['lowerGyro']['x'], data['lowerGyro']['y'], data['lowerGyro']['z']))
    upper_timestamps, upper_theta = processPair(upperAccel, upperGyro, True)
    lower_timestamps, lower_theta = processPair(lowerAccel, lowerGyro, True)
    printSummary(upper_timestamps, upper_theta, lower_timestamps, lower_theta)
    upper_timestamps = upper_timestamps.tolist()
    lower_timestamps = lower_timestamps.tolist()
    uthetaX = upper_theta[:,0].tolist()
    uthetaY = upper_theta[:,1].tolist()
    uthetaZ = upper_theta[:,2].tolist()

    lthetaX = lower_theta[:,0].tolist()
    lthetaY = lower_theta[:,1].tolist()
    lthetaZ = lower_theta[:,2].tolist()

    fusedShotID = db['lblfusedshots'].insert_one({'upperTimestamps':upper_timestamps, 'lowerTimestamps':lower_timestamps, \
        'upperTheta':{'uthetaX':uthetaX, 'uthetaY':uthetaY, 'uthetaZ':uthetaZ}, \
        'lowerTheta':{'lthetaX':lthetaX, 'lthetaY':lthetaY, 'lthetaZ':lthetaZ}, \
        'shotType':shotType,'handedness':data['handedness']}).inserted_id

    return fusedShotID

def processPair(accel, gyro, plotResults):
    reg_accel, reg_gyro = regularizeTimestamps(accel, gyro, plotResults)
    theta = tilt_correction(reg_accel, reg_gyro)
    timestamps = reg_accel[:,0]
    return timestamps, theta

#converts accelerometer readings into pitch euler angle
#accel = [timestamp, ax, ay, az]
def pitch_from_accel(accel):
    sign = 0
    math.copysign(sign, accel[2])
    return np.rad2deg(math.atan2(accel[1], math.sqrt(math.pow(accel[2],2) + math.pow(accel[3],2))))
    #return np.rad2deg(-math.atan2(accel[1], math.pow(accel[1],2) + math.pow(accel[3],2)))

#converts accelerometer readings into roll euler angle
#accel = [timestamp, ax, ay, az]
def roll_from_accel(accel):
    return np.rad2deg(math.atan2(accel[2], math.sqrt(math.pow(accel[1],2) + math.pow(accel[3],2))))
    #return np.rad2deg(math.atan2(-accel[1], accel[3]))
    #return np.rad2deg(-math.atan2(accel[2], math.pow(accel[2],2) + math.pow(accel[3],2)))

def high_pass(gyro_current, gyro_last, theta_last, alpha):
    return (1-alpha) * (theta_last + (gyro_current - gyro_last))

def low_pass(accel_current, accel_last, alpha):
    return (alpha * accel_current) + (accel_last[1] * (1 - alpha))
    #return (1 - alpha) * accel_current + alpha * theta_last

#tilt correction for a stream of accel + gyro data with timestamps
#puts accelerometer and gyroscope data into euler angles
#uses trapezoidal integration for gyroscope
#input: accel[timestamp, ax, ay,az], gyro [timestamp, gx, gy, gz]
#output: result:{delta, frames[theta[x, y, z]]} in degrees, euler angles
def tilt_correction(accel, gyro):
    lp_alpha = 0.9
    accel_len = len(accel)
    gyro_len = len(gyro)
    theta = np.zeros((accel_len, 3))
    delta = (accel[1,0] - accel[0,0]) / 1000.0
    #print(delta)

    tc = 0.5
    alpha = tc / (tc + delta)


    for index in range(accel_len):
        lowpass_accel = low_pass(accel[index,:], accel[index-1,:], alpha)
        theta[index,0] = alpha * (theta[index-1,0] + (gyro[index,1] * delta)) + (1 - alpha) * pitch_from_accel(lowpass_accel)
            #thetay = prior_thetay + gyro_integral * time_delta(ms)
        theta[index,1] = theta[index-1,1] + (gyro[index,2] * delta)
        theta[index,2] = alpha * (theta[index-1,2] + (gyro[index,3] * delta)) + (1 - alpha) * roll_from_accel(lowpass_accel)
    return theta

def printSummary(upper_timestamps, upper_pair, lower_timestamps, lower_pair):
    # Plot just to test regression
    plt.plot(upper_timestamps, upper_pair[:,0], '-',
            upper_timestamps, upper_pair[:,1], '-',
            upper_timestamps, upper_pair[:,2], '-',
            lower_timestamps, lower_pair[:,0], '-',
            lower_timestamps, lower_pair[:,1], '-',
            lower_timestamps, lower_pair[:,2], '-')
            #orig_gyr_data[:,0], orig_gyr_data[:,1], '-',
            #orig_gyr_data[:,0], orig_gyr_data[:,2], '-',
            #orig_gyr_data[:,0], orig_gyr_data[:,3], '-',)
    plt.legend(['upperX', 'upperY', 'upperZ', 'lowerX', 'lowerY', 'lowerZ'], loc='best')
    plt.show()
    #print(theta)

if __name__=='__main__':
    #loadFromFile(True)
    #fuseLabelledShots()
    loadJSONFolder()
