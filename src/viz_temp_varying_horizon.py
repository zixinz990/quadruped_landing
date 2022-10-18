import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tqdm
import copy


if not os.path.exists('../results'):
    os.mkdir('../results')


free_fall = False
plot_force = True
mf = 0.1
dt = 0.001
g = -9.81
force_scale = 0.005
video_writer = imageio.get_writer("../results/pc_video.mp4", fps=60)

data = np.genfromtxt('data_11.csv', delimiter=',')
data = data[:-15]
data = data.reshape((-1, 20))
times = data[:, 14]
plot_t = np.arange(0, times[-1], dt)
plot_data = []
for i in range(20):
    f = interp1d(times, data[:, i], kind='linear', fill_value="extrapolate")
    interp_data = f(plot_t)
    plot_data.append(interp_data)
plot_data = np.array(plot_data).transpose()
data = plot_data

if free_fall:
    free_fall_time_step = 100
    initial_state = data[0, :]
    initial_state[15:] = 0
    additional_data = []
    v = initial_state[7:9]
    this_frame_data = copy.deepcopy(initial_state)
    for i in range(free_fall_time_step):
        # v = initial_state[7:9]
        # next_v = v - np.array([0, g]) * dt
        next_v = v
        dx = (v + next_v) * dt / 2
        this_frame_data[:2] -= dx
        this_frame_data[3:5] -= dx
        this_frame_data[5:7] -= dx
        additional_data.append(this_frame_data)
        this_frame_data = copy.deepcopy(additional_data[-1])
        v = next_v
    additional_data = np.array(additional_data)
    additional_data = np.flip(additional_data, axis=0)
    data = np.concatenate((additional_data, data))


lb = 0.5
for i in tqdm.tqdm(range(data.shape[0])):
    frame = data[i]
    theta = frame[2]
    com = np.array([frame[0], frame[1]])
    front_x = com[0] + np.cos(theta) * lb / 2
    front_y = com[1] + np.sin(theta) * lb / 2
    rear_x = com[0] - np.cos(theta) * lb / 2
    rear_y = com[1] - np.sin(theta) * lb / 2
    plt.scatter(frame[0], frame[1], c='r')
    plt.plot([rear_x, front_x], [rear_y, front_y], c='b')
    plt.plot([frame[3], com[0]], [frame[4], com[1]], c='b')
    plt.plot([frame[5], com[0]], [frame[6], com[1]], c='b')
    plt.scatter(frame[3], frame[4], c='g')
    plt.scatter(frame[5], frame[6], c='b')

    plt.plot([-1, 0.5], [0, 0], c='k')  # plot the floor
    # plt.plot([-0.5, 1], [0, 0], c='k')  # plot the floor

    plt.ylim(-0.1, 2)

    plt.xlim(-1, 0.5)
    # plt.xlim(-0.5, 1)
    if plot_force:
        # if frame[14] > 0.048356899284183096:
        #     plt.plot([frame[3], frame[3] + frame[15] * force_scale], [frame[4], frame[4] + (frame[16] - mf*g) * force_scale], c='r')
        #     plt.plot([frame[5], frame[5] + frame[17] * force_scale], [frame[6], frame[6] + (frame[18] - mf*g) * force_scale], c='r')
        #     # plt.arrow(frame[3], frame[4], frame[3] + frame[15] * force_scale, frame[4] + (frame[16] - mf*g) * force_scale)
        #     # plt.arrow(frame[5], frame[6], frame[5] + frame[17] * force_scale, frame[6] + (frame[18] - mf*g) * force_scale)
        # else:
        #     plt.plot([frame[3], frame[3] + frame[15] * force_scale], [frame[4], frame[4] + frame[16] * force_scale], c='r')
        #     # plt.arrow(frame[3], frame[4], frame[3] + frame[15] * force_scale, frame[4] + frame[16] * force_scale)
        plt.plot([frame[3], frame[3] + frame[15] * force_scale], [frame[4], frame[4] + (frame[16] - mf*g) * force_scale], c='r')
        plt.plot([frame[5], frame[5] + frame[17] * force_scale], [frame[6], frame[6] + (frame[18] - mf*g) * force_scale], c='r')
    # if i == 101 or 133 or 165 or 197 or 229 or 261 or 293 or 325 or 357 or 389:
    #     file_name = '../results/' + str(i) + '.jpg'
    #     plt.savefig(file_name)
    plt.savefig("../results/temp.jpg")
    plt.cla()
    plt.clf()
    video_writer.append_data(imageio.imread('../results/temp.jpg'))
video_writer.close()
