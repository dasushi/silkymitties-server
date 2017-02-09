from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt


# Input
# - Two arrays, gyro_samples and acceleration_samples, containing Sample objects
# - Timestamps are not regularly spaced

# Output
# - Two arrays, gyro_samples and acceleration_samples, containing Sample objects
# - Timestamps are regularly spaced

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

gyro_samples = []
acceleration_samples = []

gyro_samples.append(Sample(0, 0.3, 0.5, 0.2))
gyro_samples.append(Sample(5, 0.2, 0.3, 0.1))
gyro_samples.append(Sample(9, 0.6, 0.5, 0.9))
gyro_samples.append(Sample(15, 0.9, 0.8, 0.6))

acceleration_samples.append(Sample(1, 0.7, 0.9, 0.2))
acceleration_samples.append(Sample(3, 0.2, 0.2, 0.8))
acceleration_samples.append(Sample(8, 0.1, 0.7, 0.5))
acceleration_samples.append(Sample(13, 0.7, 0.3, 0.1))


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

# Sample object property arrays
original_gyro_timestamps = extract_property_values(gyro_samples, 'timestamp')
original_gyro_x = extract_property_values(gyro_samples, 'x')
original_gyro_y = extract_property_values(gyro_samples, 'y')
original_gyro_z = extract_property_values(gyro_samples, 'z')

original_acceleration_timestamps = extract_property_values(acceleration_samples, 'timestamp')
original_acceleration_x = extract_property_values(acceleration_samples, 'x')
original_acceleration_y = extract_property_values(acceleration_samples, 'y')
original_acceleration_z = extract_property_values(acceleration_samples, 'z')

# Regression functions
interpolate_gyro_x = interp1d(original_gyro_timestamps, original_gyro_x)
interpolate_gyro_y = interp1d(original_gyro_timestamps, original_gyro_y)
interpolate_gyro_z = interp1d(original_gyro_timestamps, original_gyro_z)

interpolate_accleration_x = interp1d(original_acceleration_timestamps, original_acceleration_x)
interpolate_accleration_y = interp1d(original_acceleration_timestamps, original_acceleration_y)
interpolate_accleration_z = interp1d(original_acceleration_timestamps, original_acceleration_z)

# Create regularly-spaced timestamp array
first_timestamp = max(gyro_samples[0].timestamp, acceleration_samples[0].timestamp)
last_timestamp = min(gyro_samples[-1].timestamp, acceleration_samples[-1].timestamp)
num_samples = (len(gyro_samples) + len(acceleration_samples)) / 2.0
regularly_spaced_timestamps = np.linspace(0, last_timestamp, num_samples, endpoint=True)

regularly_spaced_gyro_x = interpolate_gyro_x(regularly_spaced_timestamps)
regularly_spaced_gyro_y = interpolate_gyro_y(regularly_spaced_timestamps)
regularly_spaced_gyro_z = interpolate_gyro_z(regularly_spaced_timestamps)

regularly_spaced_accleration_x = interpolate_accleration_x(regularly_spaced_timestamps)
regularly_spaced_accleration_y = interpolate_accleration_y(regularly_spaced_timestamps)
regularly_spaced_accleration_z = interpolate_accleration_z(regularly_spaced_timestamps)

# Empty the gyro and accleration sample arrays
gyro_samples = []
acceleration_samples = []

# Fill gyro and accleration sample arrays with regularly-spaced values
index = 0
for t in regularly_spaced_timestamps:
    gyro_samples.append(Sample(t, regularly_spaced_gyro_x[index], regularly_spaced_gyro_y[index], regularly_spaced_gyro_z[index]))
    acceleration_samples.append(Sample(t, regularly_spaced_accleration_x[index], regularly_spaced_accleration_y[index], regularly_spaced_accleration_z[index]))

# Plot just to test regression
plt.plot(original_gyro_timestamps, original_gyro_x, 'o',
    regularly_spaced_timestamps, regularly_spaced_gyro_x, '-',
    regularly_spaced_timestamps, regularly_spaced_gyro_y, '-',
    regularly_spaced_timestamps, regularly_spaced_gyro_z, '-',
    regularly_spaced_timestamps, regularly_spaced_accleration_x, '-',
    regularly_spaced_timestamps, regularly_spaced_accleration_y, '-',
    regularly_spaced_timestamps, regularly_spaced_accleration_z, '-',)
plt.legend(['original_gyro_x', 'regularly_spaced_gyro_x'], loc='best')
plt.show()
