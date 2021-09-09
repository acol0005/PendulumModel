from main import Linkage, Pendulum, animate_solution

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as integrate
import seaborn as sns
from collections import deque

if __name__ == '__main__':
    linkage_1 = Linkage(1, 1, np.pi / 2, 0)
    linkage_2 = Linkage(1, 1, np.pi / 2, 0)
    linkage_3 = Linkage(0.001, 1, 0, 0)
    modified_linkage_3 = Linkage(1, 1, np.deg2rad(1), 0)
    pendulum = Pendulum([linkage_1, linkage_2, linkage_3], 1)
    modified_pendulum = Pendulum([linkage_1, linkage_2, modified_linkage_3], 1)
    solution, df = pendulum.solve(100)
    modified_solution, modified_df = modified_pendulum.solve(100)

    fig, axs = pendulum.plot_all_linkage_variables()

    ani = animate_solution([pendulum, modified_pendulum])
    plt.show()