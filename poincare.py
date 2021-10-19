import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
import multiprocessing
import os
import time

from main import Linkage, Pendulum, animate_solution


def generate_poincare_section(pendulum,admissable_vals, x_vals, y_vals, solved=False):
    """

    :param pendulum:
    :param admissable_vals: (func) Takes a dataframe and returns a boolean mask of that dataframe, of points you want to plot in the Poincare section
    :param x_vals: (func) Takes a dataframe and returns a vector of the x-values for the Poincare section
    :param y_vals: (func) Takes a dataframe and returns a vector of the y-values for the Poincare section
    :param solved: (bool) Whether the pendulum has already been solved or not. If True, grabs df and solution from pendulum instance
    :return:
    """
    if not solved:
        solution, df = pendulum.solve()
    else:
        solution = pendulum.solution
        df = pendulum.df
    df = admissable_vals(df)
    fig, ax = plt.subplots()
    ax.scatter(x_vals(df), y_vals(df), s=1)
    return pendulum, ax

if __name__ == '__main__':
    MAX_TIME = 20000
    def get_section(df):
        spline = InterpolatedUnivariateSpline(df.time, df.theta1_deg, k=3)
        roots = spline.roots()
        # Add the new roots as indices and interpolate all the values to those new points
        df = df.append(pd.DataFrame(index=roots)).sort_index().interpolate(method='polynomial', order=1)
        df = df.loc[roots]
        mask_1 = df.omega1 > 0
        df = df.loc[mask_1]
        return df

    def get_theta3(df):
        return np.fmod(df.theta3_deg, 360)

    def get_omega3(df):
        return df.omega3_deg

    def get_theta2(df):
        return np.fmod(df.theta2_deg, 360)

    def get_omega2(df):
        return df.omega2_deg

    linkage_1 = Linkage(1, 1, 0, 0)
    linkage_2 = Linkage(1, 1, 0, 0)
    linkage_3 = Linkage(1, 1, np.pi/2, 0)
    pendulum_1 = Pendulum([linkage_1, linkage_2, linkage_3], 1, MAX_TIME)
    t0 = time.time()
    pendulum_1, ax1 = generate_poincare_section(pendulum_1, get_section, get_theta3, get_omega3)
    print(f'First Poincare diagram took {time.time() - t0:.2f}s')
    ax1.set_title(f'Poincare at Lowest Energy = {pendulum_1.calculate_energy()[0]}')
    ax1.set_xlabel('Theta 3')
    ax1.set_ylabel('Omega 3')

    pendulum_1, ax1 = generate_poincare_section(pendulum_1, get_section, get_theta2, get_omega2, solved=True)
    ax1.set_title(f'Poincare at Lowest Energy = {pendulum_1.calculate_energy()[0]}')
    ax1.set_xlabel('Theta 2')
    ax1.set_ylabel('Omega 2')

    pendulum_1.plot_all_linkage_variables()

    linkage_1 = Linkage(1, 1, 0, 0)
    linkage_2 = Linkage(1, 1, np.pi/2, 0)
    linkage_3 = Linkage(1, 1, np.pi/2, 0)
    pendulum_2 = Pendulum([linkage_1, linkage_2, linkage_3], 1, MAX_TIME)
    t0 = time.time()
    pendulum_2, ax2 = generate_poincare_section(pendulum_2, get_section, get_theta3, get_omega3)
    print(f'Second Poincare diagram took {time.time() - t0:.2f}s')
    ax2.set_title(f'Poincare at Middle Energy Level = {pendulum_2.calculate_energy()[0]}')
    ax2.set_xlabel('Theta 3')
    ax2.set_ylabel('Omega 3')

    pendulum_2, ax2 = generate_poincare_section(pendulum_2, get_section, get_theta2, get_omega2, solved=True)
    ax2.set_title(f'Poincare at Middle Energy = {pendulum_2.calculate_energy()[0]}')
    ax2.set_xlabel('Theta 2')
    ax2.set_ylabel('Omega 2')
    pendulum_2.plot_all_linkage_variables()

    linkage_1 = Linkage(1, 1, np.pi/2, 0)
    linkage_2 = Linkage(1, 1, np.pi/2, 0)
    linkage_3 = Linkage(1, 1, np.pi/2, 0)
    pendulum_3 = Pendulum([linkage_1, linkage_2, linkage_3], 1, MAX_TIME)
    t0 = time.time()
    pendulum_3, ax3 = generate_poincare_section(pendulum_3, get_section, get_theta3, get_omega3)
    print(f'Third Poincare diagram took {time.time() - t0:.2f}s')
    ax3.set_title(f'Poincare at Highest Energy Level = {pendulum_3.calculate_energy()[0]}')
    ax3.set_xlabel('Theta 3')
    ax3.set_ylabel('Omega 3')

    pendulum_3, ax3 = generate_poincare_section(pendulum_3, get_section, get_theta2, get_omega2, solved=True)
    ax3.set_title(f'Poincare at Highest Energy = {pendulum_3.calculate_energy()[0]}')
    ax3.set_xlabel('Theta 2')
    ax3.set_ylabel('Omega 2')
    pendulum_3.plot_all_linkage_variables()
    plt.show()


