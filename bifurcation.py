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
    pendulum.calculate_energy()
    return pendulum, df


def generate_bifurcation_diagram(pendulums, x_val, y_val, num_processes=os.cpu_count() - 2, solved=False, ax=None, single_valued=True):
    """
    Generates a bifurcation diagram using the supplied pendulums.

    :param pendulums: (list) List of Pendulum instances
    :param x_val: (func) Function that takes a pendulum instance as a variable and returns a number. This will be the
                  x-axis of the bifurcation diagram.
    :param y_val: (func) Function that takes a pendulum instance as a variable and returns a number. This will be the
                  y-axis of the bifurcation diagram.
    :param solved: (bool) Whether the pendulums passed have been solved or not. If False, will solve again
    :param ax: (matplotlib.axes) If an axes object is passed, scatter plot will be done on that
    :param single_valued: (bool) Whether y_vals will return a single value or multiple
    :return: (plt.axes instance) Axis of the bifurcation diagram
    """

    if not solved:
        with multiprocessing.Pool(num_processes) as pool:
            outputs = list(pool.imap(solve_pendulum, pendulums))
        # Split up the outputs
        pendulums = [x[0] for x in outputs]
        dfs = [x[1] for x in outputs]
    else:
        dfs = [pendulum.df for pendulum in pendulums]

    x_vals = np.array([x_val(df) for df in dfs])
    if not ax:
        fig, ax = plt.subplots()
    if single_valued:
        y_vals = np.array([y_val(df) for df in dfs])

        ax.scatter(x_vals, y_vals, s=1)
    else:
        y_vals = [y_val(df) for df in dfs]
        max_length = max(len(arr) for arr in y_vals)
        y_vals = np.array([np.concatenate([arr, np.ones(max_length - len(arr))*np.nan]) for arr in y_vals])
        for i in range(max_length):
            ax.scatter(x_vals, y_vals[:, i], s=1, c='cornflowerblue')
    return pendulums, ax



if __name__ == '__main__':
    def get_maximum_of_column_func(column):
        return lambda df: df[column].max() % 360

    def get_minimum_of_column_func(column):
        return lambda df: math.fmod(df[column].min(), 360)

    def get_local_maxima_func(column):
        return lambda df: df.loc[(df[column].shift(1) < df[column]) & (df[column].shift(-1) < df[column]), column] % 360

    def get_local_minima_func(column):
        return lambda df: df.loc[(df[column].shift(1) > df[column]) & (df[column].shift(-1) > df[column]), column].mod(360)



    # Sweep initial angular velocity of bars
    n = 200
    t_end = 50
    # First bar specs
    m1 = np.ones(n)
    l1 = np.ones(n)
    th1 = np.zeros(n)
    omega1 = np.linspace(np.deg2rad(0), np.pi / 3, n)
    linkages_1 = generate_list_of_linkages(m1, l1, th1, omega1)

    # Second bar specs
    m2 = np.ones(n)
    l2 = np.ones(n)
    th2 = np.zeros(n)
    omega2 = np.linspace(np.deg2rad(0), np.pi / 3, n)
    linkages_2 = generate_list_of_linkages(m2, l2, th2, omega2)

    # Second bar specs
    m3 = np.ones(n)
    l3 = np.ones(n)
    th3 = np.zeros(n)
    omega3 = np.linspace(np.deg2rad(0), np.pi / 3, n)
    linkages_3 = generate_list_of_linkages(m3, l3, th3, omega3)

    pendulums = generate_list_of_pendulums(linkages_1, linkages_2, linkages_3, 1, t_end)
    t0 = time.time()
    x_vals = lambda df: df.energy.iloc[0]

    # # Bifurcation diagram for max theta1
    # y_vals = get_maximum_of_column_func('theta1_deg')
    # pendulums, ax = generate_bifurcation_diagram(pendulums, x_vals, y_vals)
    # y_vals_2 = get_minimum_of_column_func('theta1_deg')
    # pendulums, ax = generate_bifurcation_diagram(pendulums, x_vals, y_vals_2, solved=True, ax=ax)
    # ax.grid()
    # ax.set_xlabel('Energy')
    # ax.set_ylabel('Max Theta for 1st Linkage (deg)')
    # ax.set_title('Bifurcation Diagram for Theta3_0')
    #
    # # Bifurcation diagram for max theta2
    # y_vals = get_maximum_of_column_func('theta2_deg')
    # pendulums, ax = generate_bifurcation_diagram(pendulums, x_vals, y_vals, solved=True)
    # y_vals_2 = get_minimum_of_column_func('theta2_deg')
    # pendulums, ax = generate_bifurcation_diagram(pendulums, x_vals, y_vals_2, solved=True, ax=ax)
    # ax.grid()
    # ax.set_xlabel('Energy')
    # ax.set_ylabel('Max Theta for 2nd Linkage (deg)')
    # ax.set_title('Bifurcation Diagram for Theta 2 Max Angle')
    #
    # # Bifurcation diagram for max theta3
    # y_vals = get_maximum_of_column_func('theta3_deg')
    # pendulums, ax = generate_bifurcation_diagram(pendulums, x_vals, y_vals, solved=True)
    # y_vals_2 = get_minimum_of_column_func('theta3_deg')
    # pendulums, ax = generate_bifurcation_diagram(pendulums, x_vals, y_vals_2, solved=True, ax=ax)
    # ax.grid()
    # ax.set_xlabel('Energy')
    # ax.set_ylabel('Max Theta for 3rd Linkage (deg)')
    # ax.set_title('Bifurcation Diagram for Theta 3 Max Angle')


    # Bifurcation diagram for max theta1
    y_vals = get_local_maxima_func('theta1_deg')
    pendulums, ax = generate_bifurcation_diagram(pendulums, x_vals, y_vals, single_valued=False)
    y_vals_2 = get_local_maxima_func('theta1_deg')
    pendulums, ax = generate_bifurcation_diagram(pendulums, x_vals, y_vals_2, solved=True, ax=ax, single_valued=False)
    ax.grid()
    ax.set_xlabel('Energy')
    ax.set_ylabel('Max Theta for 1st Linkage (deg)')
    ax.set_title('Bifurcation Diagram for Theta3_0')

    # Bifurcation diagram for max theta2
    y_vals = get_local_maxima_func('theta2_deg')
    pendulums, ax = generate_bifurcation_diagram(pendulums, x_vals, y_vals, solved=True, single_valued=False)
    y_vals_2 = get_local_minima_func('theta2_deg')
    pendulums, ax = generate_bifurcation_diagram(pendulums, x_vals, y_vals_2, solved=True, ax=ax, single_valued=False)
    ax.grid()
    ax.set_xlabel('Energy')
    ax.set_ylabel('Max Theta for 2nd Linkage (deg)')
    ax.set_title('Bifurcation Diagram for Theta 2 Max Angle')

    # Bifurcation diagram for max theta3
    y_vals = get_local_maxima_func('theta3_deg')
    pendulums, ax = generate_bifurcation_diagram(pendulums, x_vals, y_vals, solved=True, single_valued=False)
    y_vals_2 = get_local_maxima_func('theta3_deg')
    pendulums, ax = generate_bifurcation_diagram(pendulums, x_vals, y_vals_2, solved=True, ax=ax, single_valued=False)
    ax.grid()
    ax.set_xlabel('Energy')
    ax.set_ylabel('Max Theta for 3rd Linkage (deg)')
    ax.set_title('Bifurcation Diagram for Theta 3 Max Angle')




    print(f'Bifurcation diagram took {time.time() - t0:.2f}s')

    pendulums[int(n/4)].plot_all_linkage_variables()
    pendulums[int(n / 2)].plot_all_linkage_variables()
    pendulums[int(3*n / 4)].plot_all_linkage_variables()
    ani = animate_solution([pendulums[int(3*n / 4)]])
    plt.show()

