import numpy as np
import math

#overall:
#input: [accel[timestamp, ax, ay, az], gyro [timestamp, gx, gy, gz]]
#output: interval, theta[tx, ty, tz]
#start_time = math.max(accel[0]['timestamp'], gyro[0]['timestamp'])
#end_time = math.min(accel[-1]['timestamp'], gyro[-1]['timestamp'])
#theta.length = min(start_time, end_time)

def preprocessing(filename):


    start_time = math.max(accel[0]['timestamp'], gyro[0]['timestamp'])
    end_time = math.min(accel[-1]['timestamp'], gyro[-1]['timestamp'])
    time_length = end_time - start_time
    time_delta =
    theta.length = min(start_time, end_time)

#converts accelerometer readings into pitch euler angle
#accel = [timestamp, ax, ay, az]
def pitch_from_accel(accel):
    return np.rad2deg(-math.atan2(accel[3], math.sign(accel[2])*math.sqrt(accel[1])**2 + accel[2]**2)))

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
    index = 0
    alpha = 0.98
    theta = np.zeros(log_data.ndim, 3)

    for row in log_data:
        #need 3 samples until gyro integration via sampling (simpsons) is possible
        #fill first 2 with just gyro & accel
        if index < 2:
            theta[index,0] =  (alpha * row[4]) + (1 - alpha) * pitch_from_accel(row[1:3])
            theta[index,1] =  row[5]
            theta[index,2] =  (alpha * row[6]) + (1 - alpha) * roll_from_accel(row[1:2])
        #process from 3rd entry until the end with integration
        else:
            #x values for integration
            timestamps = [log_data[index-2,0],log_data[index-1,0],row[0]]
            #integrate x, y and z for gyroscope at this timestamp
            #done using trapezoidal method
            #TODO: experiment integrating with more samples?
            wx = np.trapz([log_data[index-2,4],log_data[index-1,4],row[4]], x=timestamps)
            wy = np.trapz([log_data[index-2,5],log_data[index-1,5],row[5]], x=timestamps)
            wz = np.trapz([log_data[index-2,6],log_data[index-1,6],row[6]], x=timestamps)
            theta[index,0] = alpha * (theta[index-1,0] * wx * (timestamp - last_timestamp)) +
                    (1 - alpha) * theta_x_atan(row[1:3])
            #thetay = prior_thetay + gyro_integral * time_delta(ms)
            theta[index,1] = theta[index-1,1] + wy * (row[0] - log_data[index-1,0])

            theta[index,2] = alpha * (theta[index-1,2] * wz * (timestamp - last_timestamp)) +
                    (1 - alpha) * roll_from_accel(row[1:2])
        index++

    return theta
