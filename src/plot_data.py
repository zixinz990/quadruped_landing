import os
import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":
    g = -9.81  # gravity

    mb = 10.0  # body mass
    mf = 0.1   # foot mass
    if not os.path.exists('../results'):
        os.mkdir('../results')

    data = np.genfromtxt('data_11.csv', delimiter=',')
    data = data[:-15]
    data = data.reshape((-1, 20))

    body_x = data[:, 0]
    body_y = data[:, 1]
    body_theta = data[:, 2]

    foot_1_x = data[:, 3]
    foot_1_y = data[:, 4]

    foot_2_x = data[:, 5]
    foot_2_y = data[:, 6]

    body_vx = data[:, 7]
    body_vy = data[:, 8]
    body_omega = data[:, 9]

    foot_1_vx = data[:, 10]
    foot_1_vy = data[:, 11]

    foot_2_vx = data[:, 12]
    foot_2_vy = data[:, 13]

    times = data[:, 14]

    F1x = data[:, 15]
    F1y = data[:, 16]
    F2x = data[:, 17]
    F2y = data[:, 18]

    GRF_1y = F1y - mf * g
    GRF_2y = F2y - mf * g

    GRF_1 = np.zeros(40)
    GRF_2 = np.zeros(40)

    for i in range(40):
        GRF_1[i] = np.sqrt(F1x[i] * F1x[i] + GRF_1y[i] * GRF_1y[i])
        GRF_2[i] = np.sqrt(F2x[i] * F2x[i] + GRF_2y[i] * GRF_2y[i])

    plt.plot(times, body_x, label="body com position x")
    plt.plot(times, body_y, label="body com position y")
    plt.ylabel("Position (m)")
    plt.xlabel("Time (sec)")
    plt.title("Body COM Position")
    plt.legend()
    plt.savefig('../results/body_pos.jpg')

    plt.cla()
    plt.clf()
    plt.plot(times, body_vx, label="body com linear velocity x")
    plt.plot(times, body_vy, label="body com linear velocity y")
    plt.ylabel("Velocity (m/s)")
    plt.xlabel("Time (sec)")
    plt.title("Body COM Linear Velocity")
    plt.legend()
    plt.savefig('../results/body_lin_vel.jpg')

    plt.cla()
    plt.clf()
    plt.plot(times, body_theta, label="body orientation")
    plt.ylabel("Orientation Angel (rad)")
    plt.xlabel("Time (sec)")
    plt.title("Body Orientation")
    plt.legend()
    plt.savefig('../results/body_orientation.jpg')

    plt.cla()
    plt.clf()
    plt.plot(times, body_omega, label="body angular velocity")
    plt.ylabel("Velocity (rad/s)")
    plt.xlabel("Time (sec)")
    plt.title("Body Angular Velocity")
    plt.legend()
    plt.savefig('../results/body_ang_vel.jpg')

    plt.cla()
    plt.clf()
    # plt.plot(times, F1x, label="GRF 1 x")
    # plt.plot(times, GRF_1y, label="GRF 1 y")
    plt.plot(times, GRF_1, label="GRF 1")
    # plt.plot(times, F2x, label="F2x")
    # plt.plot(times, F2y, label="F2y")
    # plt.axhline(y=-mf*g / 2, color='b', linestyle='--')
    plt.plot(times, GRF_2, label="GRF 2")
    plt.ylabel("Force (N)")
    plt.xlabel("Time (sec)")
    plt.title("GRFs on the Feet")
    plt.legend()
    plt.savefig('../results/force.jpg')

    plt.cla()
    plt.clf()
    plt.plot(times, foot_2_x, label='foot 2 position x')
    plt.plot(times, foot_2_y, label='foot 2 position y')
    plt.ylabel("Position (m)")
    plt.xlabel("Time (sec)")
    plt.title("Rear Foot Position")
    plt.legend()
    plt.savefig('../results/foot_position.jpg')

    plt.cla()
    plt.clf()
    plt.plot(times, foot_2_vx, label='foot 2 velocity x')
    plt.plot(times, foot_2_vy, label='foot 2 velocity y')
    plt.ylabel("Velocity (m/s)")
    plt.xlabel("Time (sec)")
    plt.title("Rear Foot Velocity")
    plt.legend()
    plt.savefig('../results/foot_velocity.jpg')
