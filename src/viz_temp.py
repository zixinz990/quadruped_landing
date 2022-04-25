import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
if not os.path.exists('../results'):
    os.mkdir('../results')
video_writer = imageio.get_writer("../results/pc_video.mp4", fps=10)

data = np.genfromtxt('data.csv', delimiter=',')
data = data[:-14]
data = data.reshape((-1,18))
lb = 1
for i in range(data.shape[0]):
    frame = data[i]
    theta = frame[2]
    com = np.array([frame[0], frame[1]])
    front_x = com[0] + np.cos(theta) * lb / 2 
    front_y = com[1] + np.sin(theta) * lb / 2 
    rear_x = com[0] - np.cos(theta) * lb / 2 
    rear_y = com[1] - np.sin(theta) * lb / 2 
    plt.scatter(frame[0], frame[1], c='r')
    plt.plot([rear_x, front_x], [rear_y, front_y])
    plt.scatter(frame[3], frame[4], c='g')
    plt.scatter(frame[5], frame[6], c='b')
    plt.plot([-1,0.5], [0,0], c='k') # plot the floor
    plt.ylim(-1,1)
    plt.xlim(-1, 0.5)
    plt.savefig('../results/temp.jpg')
    plt.cla()
    plt.clf()
    video_writer.append_data(imageio.imread('../results/temp.jpg'))
video_writer.close()

