import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import bagpy
from bagpy import bagreader   # this contains a class that does all the hard work of reading bag files

bag_name = "_2023-03-02-14-11-28.bag"   # It's easiest if all the bag files are in the same directory as this script.
b = bagreader(bag_name)   # This creates an object that we name 'b' that contains all the information from your bag file

csvfiles = []     # To avoid mixing up topics, we save each topic as an individual csv file, since some topics might have the same headers!
for t in b.topics:
    data = b.message_by_topic(t)
    csvfiles.append(data)


state = pd.read_csv(csvfiles[2])   # The topic "odom" contains all the state information we need

vicon_time = state['Time'] - b.start_time   # Here we are extracting time and subtracting the start time of the .bag file

# Position
x = state['pose.pose.position.x']
y = state['pose.pose.position.y']
z = state['pose.pose.position.z']

# Velocity
xdot = state['twist.twist.linear.x']
ydot = state['twist.twist.linear.y']
zdot = state['twist.twist.linear.z']

# Angular Velocity (w.r.t. body frames x, y, and z)
wx = state['twist.twist.angular.x']
wy = state['twist.twist.angular.y']
wz = state['twist.twist.angular.z']

# Orientation (measured as a unit quaternion)
qx = state['pose.pose.orientation.x']
qy = state['pose.pose.orientation.y']
qz = state['pose.pose.orientation.z']
qw = state['pose.pose.orientation.w']

# If you want to use Rotation, these lines might be useful
q = np.vstack((qx,qy,qz,qw)).T      # Stack the quaternions, shape -> (N,4)
rot = Rotation.from_quat(q[0,:])    # This should be familiar from the simulator

# It's often useful to save the objects associated with a figure and its axes
plt.figure(num='Position vs Time', figsize=(10, 6))

# You can plot using multiple lines if you want it to be readable
plt.plot(vicon_time, x, 'r.', markersize=2)
plt.plot(vicon_time, y, 'g.', markersize=2)
plt.plot(vicon_time, z, 'b.', markersize=2)
plt.legend(('x', 'y', 'z'), loc='upper right')   # Set a legend
plt.xlabel('time, s')                    # Set a x label
plt.ylabel('position, m')                    # Set a y label
plt.grid('major')                                # Put on a grid
plt.title('Position')                        # Plot title
plt.show()