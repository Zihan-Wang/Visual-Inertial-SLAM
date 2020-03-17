import utils
import numpy as np
import matplotlib.pyplot as plt

# load IMU inverse pose in part(c)
U = np.load('slam0034_U.npy')

# transform inverse pose to regular pose
R = np.zeros((4,4))
for i in range(U.shape[2]):
    R = np.dstack((R, np.linalg.inv(U[:,:,i])))
R = R[:,:,1:]

# load landmark poses in part(c)
markland = np.load('landmark0034.npy')
h = markland[:,-1]

# get landmarks as 2D coordinate (x, y)
hx = h[0::3]
hy = h[1::3]

# show them together
fig1, ax1 = utils.visualize_trajectory_2d(R, hx, hy, 'VI-SLAM (v=1000, w=10)')
ax1.set_title('VI-SLAM of 0034')

# load IMU inverse pose in part(a)
U = np.load('pred0034.npy')

# transform inverse pose to regular pose
R = np.zeros((4,4))
for i in range(U.shape[2]):
    R = np.dstack((R, np.linalg.inv(U[:,:,i])))
R = R[:,:,1:]

# load landmark poses in part(b)
markland = np.load('update0034.npy')
h = markland[:,-1]

# get landmarks as 2D coordinate (x, y)
hx = h[0::3]
hy = h[1::3]

# show them together
fig2, ax2 = utils.visualize_trajectory_2d(R, hx, hy,'dead-reckon')
ax2.set_title('dead-reckon of 0034')
plt.show(block=True)