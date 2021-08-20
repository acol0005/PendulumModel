from main import Linkage, Pendulum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as integrate
import seaborn as sns
from collections import deque

if __name__ == '__main__':
    linkage_1 = Linkage(1, 1, np.pi / 6, 0)
    linkage_2 = Linkage(1, 1, np.pi / 6, 0)
    linkage_3 = Linkage(1, 1, np.pi / 6, 0)
    pendulum = Pendulum([linkage_1, linkage_2, linkage_3], 1)
    solution, df = pendulum.solve(100)

    fig, axs = pendulum.plot_all_linkage_variables()

    # ani = pendulum.animate_solution(df)
    L = sum(linkage.l for linkage in pendulum.linkages)
    x1 = pendulum.linkages[0].l * np.sin(df.theta1.values)
    y1 = -pendulum.linkages[0].l * np.cos(df.theta1.values)

    x2 = pendulum.linkages[1].l * np.sin(df.theta2.values) + x1
    y2 = -pendulum.linkages[1].l * np.cos(df.theta2.values) + y1

    x3 = pendulum.linkages[2].l * np.sin(df.theta3.values) + x2
    y3 = -pendulum.linkages[2].l * np.cos(df.theta3.values) + y2


    dt = df.time.diff().mean()
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, L))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    trace1 = ax.scatter([x1[0]], [y1[0]], cmap=plt.get_cmap('Greens'), alpha=0.5)
    trace2 = ax.scatter([x2[0]], [y2[0]], cmap=plt.get_cmap('Oranges'), alpha=0.5)
    trace3 = ax.scatter([x3[0]], [y3[0]], cmap=plt.get_cmap('Blues'), alpha=0.5)

    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    history_len = 500
    history_x_1, history_y_1 = deque(maxlen=history_len), deque(maxlen=history_len)
    history_x_2, history_y_2 = deque(maxlen=history_len), deque(maxlen=history_len)
    history_x_3, history_y_3 = deque(maxlen=history_len), deque(maxlen=history_len)


    def animate(i):
        thisx = [0, x1[i], x2[i], x3[i]]
        thisy = [0, y1[i], y2[i], y3[i]]

        if i == 0:
            history_x_1.clear()
            history_y_1.clear()
            history_x_2.clear()
            history_y_2.clear()
            history_x_3.clear()
            history_y_3.clear()

        history_x_1.appendleft(thisx[1])
        history_y_1.appendleft(thisy[1])
        history_x_2.appendleft(thisx[2])
        history_y_2.appendleft(thisy[2])
        history_x_3.appendleft(thisx[3])
        history_y_3.appendleft(thisy[3])

        line.set_data(thisx, thisy)
        trace1.set_offsets(np.c_[history_x_1, history_y_1])
        trace1.set_cmap('Greens')
        trace1.set_clim(0, 1)
        trace1.set_sizes(1.0* np.ones(len(history_x_1)))
        trace1.set_array(np.linspace(1, 0, len(history_x_1)))

        trace2.set_offsets(np.c_[history_x_2, history_y_2])
        trace2.set_cmap('Oranges')
        trace2.set_clim(0, 1)
        trace2.set_sizes(1.0* np.ones(len(history_x_2)))
        trace2.set_array(np.linspace(1, 0, len(history_x_2)))

        trace3.set_offsets(np.c_[history_x_3, history_y_3])
        trace3.set_cmap('Blues')
        trace3.set_clim(0, 1)
        trace3.set_sizes(1.0*np.ones(len(history_x_3)))
        trace3.set_array(np.linspace(1, 0, len(history_x_3)))

        time_text.set_text(time_template % (i * dt))
        return line, trace1, trace2, trace3, time_text


    ani = animation.FuncAnimation(
        fig, animate, len(df), interval=dt * 500, blit=True)
    ani.save('pendulum.mp4')
    plt.show()