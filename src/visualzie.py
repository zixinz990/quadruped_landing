import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

if __name__ == "__main__":
    single_data = np.array([1, 1, 0, 0, 0, 0, 0])
    data = np.tile(single_data, (100, 1))
    if not os.path.exists('../results'):
        os.mkdir('../results')
    video_writer = imageio.get_writer("../results/pc_video.mp4", fps=10)
    
    # data = np.zeros((100, 7))
    body_length = 1
    l_body = 1.0   # body length
    l_thigh = 0.3   # thigh length
    l_calf = 0.3

    body_x = data[:, 0]
    body_y = data[:,1]
    body_theta = data[:,2]
    front_joint_0 = data[:,3]
    rear_joint_0 = data[:,4]
    front_joint_1 = data[:,5]
    rear_joint_1 = data[:,6]

    "--------------compute node point coordinates----------------"
    body_front_node_x = body_x + l_body / 2 * np.cos(body_theta)
    body_front_node_y = body_y + l_body / 2 * np.sin(body_theta)

    body_rear_node_x = body_x - l_body / 2 * np.cos(body_theta)
    body_rear_node_y = body_y - l_body / 2 * np.sin(body_theta)

    front_leg_node_x = body_front_node_x - l_thigh * np.cos(np.pi/2 - front_joint_0 + body_theta)
    front_leg_node_y = body_front_node_y - l_thigh * np.sin(np.pi/2 - front_joint_0 + body_theta)
 
    rear_leg_node_x = body_rear_node_x - l_thigh * np.cos(np.pi/2 - rear_joint_0 + body_theta)
    rear_leg_node_y = body_rear_node_y - l_thigh * np.sin(np.pi/2 - rear_joint_0 + body_theta)

    front_foot_node_x = front_leg_node_x + l_calf * np.cos(np.pi/2 + front_joint_0 - body_theta - front_joint_1)
    front_foot_node_y = front_leg_node_y - l_calf * np.sin(np.pi/2 + front_joint_0 - body_theta - front_joint_1)

    rear_foot_node_x = rear_leg_node_x + l_calf * np.cos(np.pi/2 + rear_joint_0 - body_theta - rear_joint_1)
    rear_foot_node_y = rear_leg_node_y - l_calf * np.sin(np.pi/2 + rear_joint_0 - body_theta - rear_joint_1)

    x_data = np.stack((rear_foot_node_x, rear_leg_node_x, body_rear_node_x, body_front_node_x, front_leg_node_x, front_foot_node_x), axis=1)
    y_data = np.stack((rear_foot_node_y, rear_leg_node_y, body_rear_node_y, body_front_node_y, front_leg_node_y, front_foot_node_y), axis=1)
    for i in range(data.shape[0]):
        plt.plot(x_data[i], y_data[i], c='b')
        plt.scatter(x_data[i], y_data[i], c='r')
        plt.savefig('../results/temp.jpg')
        video_writer.append_data(imageio.imread('../results/temp.jpg'))
    video_writer.close()

        # plt.scatter([body_front_node_x[i], body_rear_node_x[i]], [body_front_node_y[i], body_rear_node_y[i]], c='r')    