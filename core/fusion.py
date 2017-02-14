import numpy as np
import math
import csv
import os
from timestamps import regularizeTimestamps

#input: [accel[timestamp, ax, ay, az], gyro [timestamp, gx, gy, gz]]
#output: interval, theta[tx, ty, tz]
#start_time = math.max(accel[0]['timestamp'], gyro[0]['timestamp'])
#end_time = math.min(accel[-1]['timestamp'], gyro[-1]['timestamp'])
#theta.length = min(start_time, end_time)

def loadFromFile(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        data = list(reader)
        print(data)

def processPair(accel, gyro):
    regularAccel, regularGyro = regularizeTimestamps(accel, gyro)
    theta = tilt_correction(regularAccel, regularGyro)
    return theta

#converts accelerometer readings into pitch euler angle
#accel = [timestamp, ax, ay, az]
def pitch_from_accel(accel):
    sign = 0
    math.copysign(sign, accel[2])
    return np.rad2deg(-math.atan2(accel[3], sign*math.sqrt(math.pow(accel[1],2) + math.pow(accel[2],2))))

#converts accelerometer readings into roll euler angle
#accel = [timestamp, ax, ay, az]
def roll_from_accel(accel):
    return np.rad2deg(-math.atan2(-accel[1], accel[2]))

#tilt correction for a stream of accel + gyro data with timestamps
#puts accelerometer and gyroscope data into euler angles
#uses trapezoidal integration for gyroscope
#input: accel[timestamp, ax, ay,az], gyro [timestamp, gx, gy, gz]
#output: result:{delta, frames[theta[x, y, z]]} in degrees, euler angles
def tilt_correction(accel, gyro):
    alpha = 0.98
    accel_len = len(accel)
    gyro_len = len(gyro)
    theta = np.zeros((accel_len, 3))

    for index in range(accel_len):
        #need 3 samples until gyro integration via sampling (simpsons) is possible
        #fill first 2 with just gyro & accel
        if index < 2:
            theta[index,0] =  (alpha * gyro[index,1]) + (1 - alpha) * pitch_from_accel(accel[index,:])
            theta[index,1] =  gyro[index,2]
            theta[index,2] =  (alpha * gyro[index,2]) + (1 - alpha) * roll_from_accel(accel[index,:])
        #process from 3rd entry until the end with integration
        else:
            #x values for integration
            timestamps = [accel[0:index,0]]
            delta = accel[index,0] - accel[index-1,0]
            #integrate x, y and z for gyroscope at this timestamp
            #done using trapezoidal method
            #TODO: experiment integrating with more samples?
            wx = np.trapz([gyro[0:index,1]], x=timestamps)
            wy = np.trapz([gyro[0:index,2]], x=timestamps)
            wz = np.trapz([gyro[0:index,3]], x=timestamps)
            theta[index,0] = alpha * (theta[index-1,0] * wx * delta) + (1 - alpha) * pitch_from_accel(accel[index,:])
            #thetay = prior_thetay + gyro_integral * time_delta(ms)
            theta[index,1] = theta[index-1,1] + wy * (accel[index,0] - accel[index-1,0])
            theta[index,2] = alpha * (theta[index-1,2] * wz * delta) + (1 - alpha) * roll_from_accel(accel[index,:])

    return theta

with open("data/Board2_ACC_2.csv") as f:
    reader = csv.reader(f)
    acc_data = np.array(list(reader))
    acc_data = np.delete(acc_data, [0,1], axis=0)
with open("data/Board2_GYRO_2.csv") as f:
    reader = csv.reader(f)
    gyr_data = list(reader)
    gyr_data = np.delete(gyr_data, [0,1], axis=0)
theta = processPair(acc_data, gyr_data)
print(theta)
