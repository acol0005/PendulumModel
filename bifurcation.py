import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import os
import time
import math

from main import Linkage, Pendulum, animate_solution


def generate_list_of_linkages(masses, lengths, initial_angles, initial_angular_velocities):
    if not all(len(x) == len(masses) for x in (lengths, initial_angles, initial_angular_velocities)):
        raise ValueError('Not all argument lists are the same length')

    def instantiate_linkage(i):
        return Linkage(masses[i], lengths[i], initial_angles[i], initial_angular_velocities[i])

    return [instantiate_linkage(i) for i in range(len(masses))]


def generate_list_of_pendulums(linkages_1, linkages_2, linkages_3, g, t_end):
    if not all(len(x) == len(linkages_1) for x in (linkages_2, linkages_3)):
        raise ValueError('Not all argument lists are the same length')
    return [Pendulum([linkages_1[i], linkages_2[i], linkages_3[i]], g, t_end) for i in range(len(linkages_1))]


# have to define top-level func so it's pickle-able and we can multiprocess it
def solve_pendulum(pendulum):
    _, df = pendulum.solve()
    return pendulum, df


def generate_bifurcation_diagram(pendulums, x_val, y_val, num_processes=os.cpu_count() - 2):
    """
    Generates a bifurcation diagram using the supplied pendulums.

    :param pendulums: (list) List of Pendulum instances
    :param x_val: (func) Function that takes a pendulum instance as a variable and returns a number. This will be the
                  x-axis of the bifurcation diagram.
    :param y_val: (func) Function that takes a pendulum instance as a variable and returns a number. This will be the
                  y-axis of the bifurcation diagram.
    :return: (plt.axes instance) Axis of the bifurcation diagram
    """

    with multiprocessing.Pool(num_processes) as pool:
        outputs = list(pool.imap(solve_pendulum, pendulums))
    # Split up the outputs
    pendulums = [x[0] for x in outputs]
    dfs = [x[1] for x in outputs]

    x_vals = np.array([x_val(df) for df in dfs])
    y_vals = np.array([y_val(df) for df in dfs])
    fig, ax = plt.subplots()
    ax.scatter(x_vals, y_vals, s=5)
    return pendulums, ax


if __name__ == '__main__':
    # Sweep initial angle of third bar
    n = 100
    t_end = 50
    # First bar specs
    m1 = np.ones(n)
    l1 = np.ones(n)
    th1 = np.linspace(np.deg2rad(0), np.pi, n)
    omega1 = np.zeros(n)
    linkages_1 = generate_list_of_linkages(m1, l1, th1, omega1)

    # Second bar specs
    m2 = np.ones(n)
    l2 = np.ones(n)
    th2 = np.linspace(np.deg2rad(0), np.pi, n)
    omega2 = np.zeros(n)
    linkages_2 = generate_list_of_linkages(m2, l2, th2, omega2)

    # Second bar specs
    m3 = np.ones(n)
    l3 = np.ones(n)
    th3 = np.linspace(np.deg2rad(0), np.pi, n)
    omega3 = np.zeros(n)
    linkages_3 = generate_list_of_linkages(m3, l3, th3, omega3)

    pendulums = generate_list_of_pendulums(linkages_1, linkages_2, linkages_3, 1, t_end)
    t0 = time.time()
    x_vals = lambda df: df.theta3_deg.iloc[0]
    y_vals = lambda df: df.theta1_deg.max() % 360
    pendulums, ax = generate_bifurcation_diagram(pendulums, x_vals, y_vals)
    y_vals_2 = lambda df: math.fmod(df.theta1_deg.min(), 360)
    ax.scatter([x_vals(pendulum.df) for pendulum in pendulums], [y_vals_2(pendulum.df) for pendulum in pendulums], s=5)
    ax.grid()
    ax.set_xlabel('Theta0 for 3rd Linkage (deg)')
    ax.set_ylabel('Max Theta for 1st Linkage (deg')
    ax.set_title('Bifurcation Diagram for Theta3_0')
    print(f'Bifurcation diagram took {time.time() - t0:.2f}s')

    pendulums[50].plot_all_linkage_variables()
    plt.show()

