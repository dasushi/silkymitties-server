import numpy as np
import math

#overall:
#input: [timestamp, accel[ax, ay,az], gyro [gx, gy, gz]]
#output: interval, q[qa, qb, qc, qd] or [timestamp, theta[tx, ty, tz]]

#tilt correction for a stream of accel + gyro data with timestamps
#puts accelerometer and gyroscope data into euler angles
#uses trapezoidal integration for gyroscope
#input: log_data: [timestamp, accel[ax, ay,az], gyro [gx, gy, gz]]
#output: theta[x, y, z] in degrees, euler angles, same length as input log
def tilt_correction(log_data):
    index = 2
    alpha = 0.98
    theta = np.zeros(log_data.ndim, 3)
    #need 3 samples until gyro integration via sampling (simpsons) is possible
    #fill first 2 with just gyro & accel
    theta[0,0] =  (alpha * log_data[0,4]) + (1 - alpha) * np.rad2deg(-math.atan2(log_data[0,3], math.sign(log_data[0,2])*math.sqrt(log_data[0,1]**2 + log_data[0,2]**2)))
    theta[0,1] =  log_data[0,5]
    theta[0,2] =  (alpha * log_data[0,6]) + (1 - alpha) * np.rad2deg(-math.atan2(-log_data[0,1], log_data[0,2]))
    theta[1,0] =  (alpha * log_data[1,4]) + (1 - alpha) * np.rad2deg(-math.atan2(log_data[1,3], math.sign(log_data[1,2])*math.sqrt(log_data[1,1]**2 + log_data[1,2]**2)))
    theta[1,1] =  log_data[1,5]
    theta[1,2] =  (alpha * log_data[1,6]) + (1 - alpha) * np.rad2deg(-math.atan2(-log_data[1,1], log_data[1,2]))
    #process from 3rd entry until the end with integration
    for row in log_data[2,:]:
        #x values for integration
        timestamps = [log_data[index-2,0],log_data[index-1,0],row[0]]
        #integrate x, y and z for gyroscope at this timestamp
        wx = np.trapz([log_data[index-2,4],log_data[index-1,4],row[4]], x=timestamps)
        wy = np.trapz([log_data[index-2,5],log_data[index-1,5],row[5]], x=timestamps)
        wz = np.trapz([log_data[index-2,6],log_data[index-1,6],row[6]], x=timestamps)
        theta[index,0] = alpha * (theta[index-1,0] * wx * (timestamp - last_timestamp)) +
                (1 - alpha) * np.rad2deg(-math.atan2(row[3], math.sign(row[2])*math.sqrt(row[1]**2 + row[2]**2)))

        theta[index,1] = theta[index-1,1] + wy * (timestamp - last_timestamp)

        theta[index,2] = alpha * (theta[index-1,2] * wz * (timestamp - last_timestamp)) +
                (1 - alpha) * np.rad2deg(-math.atan2(-row[1], row[2]))
        index++

    return theta
