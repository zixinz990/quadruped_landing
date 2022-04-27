import os
import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":
    g = -9.81  # gravity   

    mb = 10.0  # body mass
    if not os.path.exists('../results'):
        os.mkdir('../results')

    data = np.genfromtxt('data_optimal_solution_found_1.csv', delimiter=',')
    data = data[:-15]
    data = data.reshape((-1, 20))
    times = data[:,14]

    F1x = data[:, 15]
    F1y = data[:, 16]
    F2x = data[:, 17]
    F2y = data[:, 18]

    feet_2_y = data[:, 6]
    feet_2_x = data[:, 5]
    feet_2_v = data[:, 12:14]
    feet_2_v = np.linalg.norm(feet_2_v, axis=1)
    

    plt.plot(times, F1x, label="F1x")
    plt.plot(times, F1y, label="F1y")
    plt.plot(times, F2x, label="F2x")
    plt.plot(times, F2y, label="F2y")
    plt.axhline(y=-mb*g / 2, color='r', linestyle='--')
    plt.ylabel("Force")
    plt.xlabel("Time")
    plt.title("Ground Reaction Force")
    plt.legend()
    plt.savefig('../results/force.jpg')

    plt.cla()
    plt.clf()
    plt.plot(times, feet_2_x, label='feet 2 x')
    plt.plot(times, feet_2_y, label='feet 2 y')
    plt.ylabel("Coordinate")
    plt.xlabel("Time")
    plt.title("Rear foot coordinate")
    plt.legend()
    plt.savefig('../results/foot_coordinate.jpg')

    plt.cla()
    plt.clf()
    plt.plot(times, feet_2_v, label='feet 2 x')
    plt.ylabel("Velocity")
    plt.xlabel("Time")
    plt.title("Rear foot velocity")
    plt.legend()
    plt.savefig('../results/foot_velocity.jpg')

