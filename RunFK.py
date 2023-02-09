import ForwardKinematics as FK

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from tqdm import tqdm

arm = FK.BarrettArm()

jointdatafile = open('JointData.txt', 'r')
lines = jointdatafile.readlines()
eePositions = np.zeros((len(lines), 3))

for i, thetasStr in tqdm(enumerate(lines)):
    thetas = [float(theta) for theta in thetasStr.split(" ")]
    eePositions[i,:] = arm.get_ee_position(thetas)

# Create output text file with end effector positions
eePositionsStr = [' '.join([str(v) for v in line]) for line in eePositions]
markerTraj = open('MarkerTipTrajectory.txt', 'w')
print("Writing end effector positions to text file")
for eePositionStr in tqdm(eePositionsStr):
    markerTraj.write(eePositionStr)
    markerTraj.write("\n")
markerTraj.close()


plt.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(5, 6))
ax = fig.add_subplot(projection='3d')
plt.subplots_adjust(bottom=0.35, top=1.0)
maxZ = 2.3
minZ = -.1
maxX = 1.2
minX = .1
maxY = 1.2
minY = 0.5
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.view_init(azim=30)

xlines = np.array(eePositions[:,0])
ylines = np.array(eePositions[:,1])
zlines = np.array(eePositions[:,2])
ax.plot3D(xlines, ylines, zlines, 'red')

plt.show()



f = open("MarkerTipTrajectory.txt", "r")
lines = f.readlines()

eePositions = np.array([[float(v) for v in line.split(" ")] for line in lines])

randids = np.random.choice(np.arange(len(eePositions)), 3, replace=False)

a = eePositions[randids[0]]
b = eePositions[randids[1]]
c = eePositions[randids[2]]


norm = np.cross(b - a, c - a) * -1
normalized = np.linalg.norm(norm)
print(f"norm: {norm / normalized}")


fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.view_init(azim=30)

# ax.scatter([.2, .2], [1.15, 1.15], [1.26, 1.5], s=0)
# ax.axis([.1, .35, .95, 1.25])

xlines = eePositions[:,0]
ylines = eePositions[:,1]
zlines = eePositions[:,2]
ax.plot3D(xlines, ylines, zlines, 'chartreuse')

a_start = a
a_end = a + norm * 1
# ax.plot3D((a_start[0], a_end[0]), (a_start[1], a_end[1]), (a_start[2], a_end[2]), 'red')

# ax.scatter([a[0], b[0], c[0]], [a[1], b[1], c[1]], [a[2], b[2], c[2]], s=100)

plt.show()