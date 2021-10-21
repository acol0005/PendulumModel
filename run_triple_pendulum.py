from main import Linkage, Pendulum, animate_solution

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as integrate
import seaborn as sns
from collections import deque

def calculate_fixed_point_energy_levels():
    linkage1_thetas = [0, np.pi]
    linkage2_thetas = [0, np.pi]
    linkage3_thetas = [0, np.pi]

    for th1 in linkage1_thetas:
        for th2 in linkage2_thetas:
            for th3 in linkage3_thetas:
                linkage_1 = Linkage(1, 1, th1, 0)
                linkage_2 = Linkage(1, 1, th2, 0)
                linkage_3 = Linkage(1, 1, th3, 0)
                pendulum = Pendulum([linkage_1, linkage_2, linkage_3], 1, 20)
                solution, df = pendulum.solve()
                energy = pendulum.calculate_energy().mean()
                print(f'E = {energy:.3f} for ICs: ({th1:.2f}, {th2:.2f}, {th3:.2f})')


if __name__ == '__main__':
    # linkage_1 = Linkage(1, 1, np.pi / 4, 0)
    # linkage_2 = Linkage(1, 1, np.pi / 4, 0)
    # linkage_3 = Linkage(1, 1, np.pi / 4, 0)
    # modified_linkage_3 = Linkage(1, 1, np.deg2rad(1), 0)
    # pendulum = Pendulum([linkage_1, linkage_2, linkage_3], 1)
    # modified_pendulum = Pendulum([linkage_1, linkage_2, modified_linkage_3], 1)
    # solution, df = pendulum.solve(100)
    # modified_solution, modified_df = modified_pendulum.solve(100)
    #
    # fig, axs = pendulum.plot_all_linkage_variables()
    #
    # ani = animate_solution([pendulum])
    # plt.show()
    calculate_fixed_point_energy_levels()