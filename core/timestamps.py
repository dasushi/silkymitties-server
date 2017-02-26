from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt


# Input
# - Two arrays, gyro_samples and acc_samples, containing Sample objects
# - Timestamps are not reg spaced

# Output
# - Two arrays, gyro_samples and acc_samples, containing Sample objects
# - Timestamps are reg spaced

# Mock data
class Sample(object):
    timestamp = 0
    x = []
    y = []
    z = []

    def __init__(self, timestamp, x, y, z):
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z

def addTestData(acc_samples, gyro_samples):
    gyro_samples.append(Sample(0, 0.3, 0.5, 0.2))
    gyro_samples.append(Sample(5, 0.2, 0.3, 0.1))
    gyro_samples.append(Sample(9, 0.6, 0.5, 0.9))
    gyro_samples.append(Sample(15, 0.9, 0.8, 0.6))

    acc_samples.append(Sample(1, 0.7, 0.9, 0.2))
    acc_samples.append(Sample(3, 0.2, 0.2, 0.8))
    acc_samples.append(Sample(8, 0.1, 0.7, 0.5))
    acc_samples.append(Sample(13, 0.7, 0.3, 0.1))
    return acc_samples, gyro_samples

def extract_property_values(sample_list, sample_property):
    property_values = []
    for sample in sample_list:
        if sample_property == 'timestamp':
            property_values.append(sample.timestamp)
        elif sample_property == 'x':
            property_values.append(sample.x)
        elif sample_property == 'y':
            property_values.append(sample.y)
        elif sample_property == 'z':
            property_values.append(sample.z)
    return property_values

def regularizeTimestamps(accel, gyro, plotResults):

    #gyro_samples = []
    #acc_samples = []

    # Sample object property arrays
    #original_gyro_timestamps = extract_property_values(gyro_samples, 'timestamp')
    #original_gyro_x = extract_property_values(gyro_samples, 'x')
    #original_gyro_y = extract_property_values(gyro_samples, 'y')
    #original_gyro_z = extract_property_values(gyro_samples, 'z')

    #original_acc_timestamps = extract_property_values(acc_samples, 'timestamp')
    #original_acc_x = extract_property_values(acc_samples, 'x')
    #original_acc_y = extract_property_values(acc_samples, 'y')
    #original_acc_z = extract_property_values(acc_samples, 'z')

    #Split input into individual arrays

    original_gyro_timestamps = np.array([(int(row[0]) - gyro[0][0]) for row in gyro])
    original_gyro_x = np.array([row[1] for row in gyro])
    original_gyro_y = np.array([row[2] for row in gyro])
    original_gyro_z = np.array([row[3] for row in gyro])

    original_acc_timestamps = np.array([(int(row[0]) - accel[0][0]) for row in accel])
    original_acc_x = np.array([row[1] for row in accel])
    original_acc_y = np.array([row[2] for row in accel])
    original_acc_z = np.array([row[3] for row in accel])

    # Regression functions
    interpolate_gyro_x = interp1d(original_gyro_timestamps, original_gyro_x)
    interpolate_gyro_y = interp1d(original_gyro_timestamps, original_gyro_y)
    interpolate_gyro_z = interp1d(original_gyro_timestamps, original_gyro_z)

    interpolate_acc_x = interp1d(original_acc_timestamps, original_acc_x)
    interpolate_acc_y = interp1d(original_acc_timestamps, original_acc_y)
    interpolate_acc_z = interp1d(original_acc_timestamps, original_acc_z)

    # Create regularly-spaced timestamp array
    first_timestamp = max(original_gyro_timestamps[0], original_acc_timestamps[0])
    last_timestamp = min(original_gyro_timestamps[-1], original_acc_timestamps[-1])
    num_samples = int(round((len(original_gyro_timestamps) + len(original_acc_timestamps)) / 2))
    reg_spaced_timestamps = np.linspace(0, last_timestamp - first_timestamp, num_samples, endpoint=True, dtype='int16')

    reg_spaced_gyro_x = interpolate_gyro_x(reg_spaced_timestamps)
    reg_spaced_gyro_y = interpolate_gyro_y(reg_spaced_timestamps)
    reg_spaced_gyro_z = interpolate_gyro_z(reg_spaced_timestamps)

    reg_spaced_acc_x = interpolate_acc_x(reg_spaced_timestamps)
    reg_spaced_acc_y = interpolate_acc_y(reg_spaced_timestamps)
    reg_spaced_acc_z = interpolate_acc_z(reg_spaced_timestamps)

    # Fill gyro and acc sample arrays with reg-spaced values
    gyro_samples = np.column_stack((reg_spaced_timestamps, reg_spaced_gyro_x, reg_spaced_gyro_y, reg_spaced_gyro_z))
    acc_samples = np.column_stack((reg_spaced_timestamps, reg_spaced_acc_x, reg_spaced_acc_y, reg_spaced_acc_z))
    #index = 0
    #for t in reg_spaced_timestamps:
    #    gyro_samples.append([t, reg_spaced_gyro_x[index], reg_spaced_gyro_y[index], reg_spaced_gyro_z[index]])
    #    acc_samples.append([t, reg_spaced_acc_x[index], reg_spaced_acc_y[index], reg_spaced_acc_z[index]])
    if(plotResults):
        # Plot just to test regression
        plt.plot(original_gyro_timestamps, original_gyro_x, 'o',
            reg_spaced_timestamps, reg_spaced_gyro_x, '-',
            reg_spaced_timestamps, reg_spaced_gyro_y, '-',
            reg_spaced_timestamps, reg_spaced_gyro_z, '-',
            reg_spaced_timestamps, reg_spaced_acc_x, '-',
            reg_spaced_timestamps, reg_spaced_acc_y, '-',
            reg_spaced_timestamps, reg_spaced_acc_z, '-',)
        plt.legend(['original_gyro_x', 'reg_spaced_gyro_x'], loc='best')
        plt.show()
    #print(acc_samples)
    #print(gyro_samples)
    return acc_samples, gyro_samples
