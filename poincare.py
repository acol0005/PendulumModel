import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
import multiprocessing
import os
import time

from main import Linkage, Pendulum, animate_solution, wrap_to_180



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
        # return np.fmod(df.theta3_deg, 360)
        return wrap_to_180(df.theta3_deg)

    def get_omega3(df):
        return df.omega3_deg

    def get_theta2(df):
        # return np.fmod(df.theta2_deg, 360)
        return wrap_to_180(df.theta2_deg)

    def get_omega2(df):
        return df.omega2_deg

    def generate_poincare_at_energy_level(th1, th2, th3):
        linkage_1 = Linkage(1, 1, th1, 0)
        linkage_2 = Linkage(1, 1, th2, 0)
        # linkage_3 = Linkage(1, 1, th3, 0.13)
        linkage_3 = Linkage(1, 1, th3, 0)
        pendulum = Pendulum([linkage_1, linkage_2, linkage_3], 1, MAX_TIME)
        t0 = time.time()
        pendulum, ax = generate_poincare_section(pendulum, get_section, get_theta3, get_omega3)
        print(f'Poincare diagram took {time.time() - t0:.2f}s')
        ax.set_title(f'Poincare at E = {pendulum.calculate_energy()[0]:.2f}')
        ax.set_xlabel('Theta 3')
        ax.set_ylabel('Omega 3')
        ax.grid()
        fig = ax.get_figure()
        fig.savefig(f'E{pendulum.calculate_energy()[0]:.2f}_Th3.pdf')

        pendulum, ax = generate_poincare_section(pendulum, get_section, get_theta2, get_omega2, solved=True)
        ax.set_title(f'Poincare at E = {pendulum.calculate_energy()[0]:.2f}')
        ax.set_xlabel('Theta 2')
        ax.set_ylabel('Omega 2')
        ax.grid()
        fig = ax.get_figure()
        fig.savefig(f'E{pendulum.calculate_energy()[0]:.2f}_Th2.pdf')

        fig, axs = pendulum.plot_all_linkage_variables()
        fig.suptitle(f'Linkage Variables at E = {pendulum.calculate_energy()[0]:.2f}')
        return pendulum, ax

    initial_thetas = [[0, 0, 0], [0, 0, np.pi], [0, np.pi, 0], [0, np.pi, np.pi], [np.pi, 0, 0]]
    initial_thetas = np.linspace(0.01, np.pi/2, 5)
    print(initial_thetas)
    # for theta in initial_thetas:
    #     generate_poincare_at_energy_level(theta.item(), theta.item(), theta.item())
    # plt.show()


