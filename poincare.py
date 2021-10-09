import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import os
import time

from main import Linkage, Pendulum, animate_solution


def generate_poincare_section(pendulum,admissable_vals, x_vals, y_vals):
    solution, df = pendulum.solve()
    df = df.loc[admissable_vals(df)]
    fig, ax = plt.subplots()
    ax.scatter(x_vals(df), y_vals(df), s=5)
    return pendulum, ax

if __name__ == '__main__':
    def get_section(df):
        mask_1 = df.omega1 > 0
        # mask_2 = df.omega2 > 0
        mask_3 = (df.theta1_deg % 360).abs() < 1
        # mask_4 = (df.theta2_deg % 360).abs() < 1
        # return mask_1 & mask_2 & mask_3 & mask_4
        return mask_1 & mask_3
    def x_vals(df):
        return df.theta3_deg % 360

    def y_vals(df):
        return df.omega3_deg % 360

    linkage_1 = Linkage(1, 1, 0, 0)
    linkage_2 = Linkage(1, 1, 0, 0)
    linkage_3 = Linkage(1, 1, np.pi/2, 0)
    pendulum_1 = Pendulum([linkage_1, linkage_2, linkage_3], 1, 1000)
    t0 = time.time()
    pendulum_1, ax1 = generate_poincare_section(pendulum_1, get_section, x_vals, y_vals)
    print(f'First Poincare diagram took {time.time() - t0:.2f}s')
    ax1.set_title('Poincare at Lowest Energy Level')
    ax1.set_xlabel('Theta 3')
    ax1.set_xlabel('Omega 3')
    pendulum_1.plot_all_linkage_variables()

    linkage_1 = Linkage(1, 1, 0, 0)
    linkage_2 = Linkage(1, 1, np.pi/2, 0)
    linkage_3 = Linkage(1, 1, np.pi/2, 0)
    pendulum_2 = Pendulum([linkage_1, linkage_2, linkage_3], 1, 1000)
    t0 = time.time()
    pendulum_2, ax2 = generate_poincare_section(pendulum_2, get_section, x_vals, y_vals)
    print(f'Second Poincare diagram took {time.time() - t0:.2f}s')
    ax2.set_title('Poincare at Middle Energy Level')
    ax2.set_xlabel('Theta 3')
    ax2.set_xlabel('Omega 3')
    pendulum_2.plot_all_linkage_variables()

    linkage_1 = Linkage(1, 1, np.pi/2, 0)
    linkage_2 = Linkage(1, 1, np.pi/2, 0)
    linkage_3 = Linkage(1, 1, np.pi/2, 0)
    pendulum_3 = Pendulum([linkage_1, linkage_2, linkage_3], 1, 1000)
    t0 = time.time()
    pendulum_3, ax3 = generate_poincare_section(pendulum_3, get_section, x_vals, y_vals)
    print(f'Third Poincare diagram took {time.time() - t0:.2f}s')
    ax3.set_title('Poincare at Highest Energy Level')
    ax3.set_xlabel('Theta 3')
    ax3.set_xlabel('Omega 3')
    pendulum_3.plot_all_linkage_variables()
    plt.show()


