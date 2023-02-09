import ForwardKinematics as FK

import modern_robotics as mr
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.widgets import Slider, Button
from tqdm import tqdm


# Create 3D plot
plt.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(5, 6))
ax = fig.add_subplot(projection='3d')
plt.subplots_adjust(bottom=0.35, top=1.0)
maxZ = 1.7
minZ = 0.5
maxX = 1.2
minX = .1
maxY = 1.2
minY = 0.5
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.view_init(azim=30)

# Define visual update functions
def drawEverything(jointPositions):
    ax.clear()
    ax.scatter([0, 0], [0, 0], [maxZ, minZ], s=0)
    plotPointsLines(jointPositions)
    # drawTrail(jointPositions)
    ax.axis([minX, maxX, minY, maxY])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.canvas.draw_idle()

def plotPointsLines(jointPositions):
    # jointPositions = np.insert(jointPositions, 0, [0, 0, 0], axis=0)
    jointPositions = np.hstack([np.array([0, 0, 0]).reshape(3,1), jointPositions])
    # Plot points
    ax.scatter(jointPositions[0,:], jointPositions[1,:], jointPositions[2,:], color='.3')
    for i, row in enumerate(jointPositions.T):
        if i == 0:
            ax.text(row[0], row[1], row[2], 'S', fontsize=10, color='red')
        elif i == len(jointPositions.T)-1:
            ax.text(row[0], row[1], row[2], 'T', fontsize=10, color='red')
        else:
            ax.text(row[0], row[1], row[2], i-1, fontsize=10, color='red')
    # Plot lines
    xlines = np.array(jointPositions[0,:])
    ylines = np.array(jointPositions[1,:])
    zlines = np.array(jointPositions[2,:])
    ax.plot3D(xlines, ylines, zlines, 'chartreuse')
    

# Instantiate arm object and initialize drawing at home position
arm = FK.BarrettArm()
jointPositions = arm.get_joint_positions([0]*8)
drawEverything(jointPositions)

# # Create joint angle sliders
# theta0 = fig.add_axes([0.25, 0.1, 0.65, 0.03])
# theta1 = fig.add_axes([0.25, 0.15, 0.65, 0.03])
# theta2 = fig.add_axes([0.25, 0.2, 0.65, 0.03])
# theta3 = fig.add_axes([0.25, 0.25, 0.65, 0.03])
# theta4 = fig.add_axes([0.25, 0.3, 0.65, 0.03])
# theta5 = fig.add_axes([0.25, 0.35, 0.65, 0.03])
# theta6 = fig.add_axes([0.25, 0.4, 0.65, 0.03])

# theta0Slider = Slider(ax=theta0,
#                       label='theta 0 [rad]',
#                       valmin=-np.pi,
#                       valmax=np.pi,
#                       valinit=0)
# theta1Slider = Slider(ax=theta1,
#                       label='theta 1 [rad]',
#                       valmin=-np.pi,
#                       valmax=np.pi,
#                       valinit=0)
# theta2Slider = Slider(ax=theta2,
#                       label='theta 2 [rad]',
#                       valmin=-np.pi,
#                       valmax=np.pi,
#                       valinit=0)
# theta3Slider = Slider(ax=theta3,
#                       label='theta 3 [rad]',
#                       valmin=-np.pi,
#                       valmax=np.pi,
#                       valinit=0)
# theta4Slider = Slider(ax=theta4,
#                       label='theta 4 [rad]',
#                       valmin=-np.pi,
#                       valmax=np.pi,
#                       valinit=0)
# theta5Slider = Slider(ax=theta5,
#                       label='theta 5 [rad]',
#                       valmin=-np.pi,
#                       valmax=np.pi,
#                       valinit=0)
# theta6Slider = Slider(ax=theta6,
#                       label='theta 6 [rad]',
#                       valmin=-np.pi,
#                       valmax=np.pi,
#                       valinit=0)

# Create button to start drawing from txt file with joint angles
startDrawingAx = fig.add_axes([0.25, 0.5, 0.1, 0.06])

startDrawingButton = Button(startDrawingAx, "start")

# # Define slider update function
# def updateTheta(val):
#     print(f"theta 0: {theta0Slider.val}\n\
# theta 1: {theta1Slider.val}\n\
# theta 2: {theta2Slider.val}\n\
# theta 3: {theta3Slider.val}\n\
# theta 4: {theta4Slider.val}\n\
# theta 5: {theta5Slider.val}\n\
# theta 6: {theta6Slider.val}\n")
#     thetas = [theta0Slider.val, theta1Slider.val, theta2Slider.val, theta3Slider.val, theta4Slider.val, theta5Slider.val, theta6Slider.val]
#     jointPositions = arm.get_joint_positions(thetas)
#     drawEverything(jointPositions)
#     print(f"new joint positions:\n {jointPositions}\n\n\n")

# Define button update function
def drawFromFile(event):
    jointdatafile = open('JointData.txt', 'r')
    lines = jointdatafile.readlines()[::50]
    allJointPositions = np.zeros((len(lines), 3, 9))
    eePositions = np.zeros((len(lines), 3))
    print("Precomputing end effector positions")
    for i, thetasStr in tqdm(enumerate(lines)):
        thetas = [float(theta) for theta in thetasStr.split(" ")]
        allJointPositions[i,:,:] = arm.get_joint_positions(thetas)
        eePositions[i,:] = allJointPositions[-1, :, -1]
    
    # Animate arm
    for i, jointPositions in enumerate(allJointPositions):
        drawEverything(jointPositions)
    
        if len(allJointPositions.shape) == 3:
            xlines = np.array(allJointPositions[:i,0,-1])
            ylines = np.array(allJointPositions[:i,1,-1])
            zlines = np.array(allJointPositions[:i,2,-1])
            ax.plot3D(xlines, ylines, zlines, 'red')
        plt.pause(0.001)
    
# # Assign sliders to a callback
# theta0Slider.on_changed(updateTheta)
# theta1Slider.on_changed(updateTheta)
# theta2Slider.on_changed(updateTheta)
# theta3Slider.on_changed(updateTheta)
# theta4Slider.on_changed(updateTheta)
# theta5Slider.on_changed(updateTheta)
# theta6Slider.on_changed(updateTheta)

# Assign button to a callback
startDrawingButton.on_clicked(drawFromFile)

plt.show()