import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as integrate
import seaborn as sns
from collections import deque


class Linkage:
    def __init__(self, mass_ball, length, initial_angle, initial_angular_velocity, mass_arm=0):
        """
        Class representing a single linkage of a pendulum.
        :param mass_ball: (float) Mass at the end of the pendulum
        :param length: (float) Length of the pendulum
        :param initial_angle: (float) Initial angle of the pendulum from the downwards vertical, positive is counter-clockwise
        :param initial_angular_velocity: (float) Initial angular velocity  positive is counter-clockwise
        :param mass_arm: (float) Mass of the arm. Assume evenly distributed
        """
        self.m = mass_ball
        self.m_arm = mass_arm
        self.l = length
        self.theta = initial_angle
        self.omega = initial_angular_velocity


class Pendulum:
    def __init__(self, linkage_list, g):
        self.linkages = linkage_list
        self.g = g
        self.solution = None
        self.df = None

    def __len__(self):
        return len(self.linkages)

    def _get_double_pendulum_rhs(self):
        """
                Currently only works for 2 pendulum model
                :return:
                """

        l1 = self.linkages[0].l
        l2 = self.linkages[1].l
        m1 = self.linkages[0].m
        m2 = self.linkages[1].m

        def rhs_func(t, y):
            # Second-order ODE means that we'll get two DEs per linkage
            rhs = np.zeros(len(self) * 2)
            theta1 = y[0]
            theta2 = y[1]
            omega1 = y[2]
            omega2 = y[3]
            rhs[0] = omega1
            rhs[1] = omega2

            # Define omega_dot_1
            numerator_1 = np.sin(theta1 - theta2) * (l1 * np.cos(theta1 - theta2) * omega1 ** 2 + omega2 ** 2)
            denominator_1 = 2 * l1 * (1 + m1 - np.cos(theta1 - theta2) ** 2)

            numerator_2 = (1 + 2 * m1) * np.sin(theta1) + np.sin(theta1 - 2 * theta2)
            denominator_2 = l1 * (1 + m1 - np.cos(theta1 - theta2) ** 2)

            rhs[2] = numerator_1 / denominator_1 - numerator_2 / denominator_2

            # Define omega_dot_2
            numerator_1 = np.sin(theta1 - theta2)
            numerator_2 = (1 + m1) * (np.cos(theta1) + l1 * omega1 ** 2) + np.cos(theta1 - theta2) * omega2 ** 2
            denominator_1 = 1 + m1 - np.cos(theta1 - theta2) ** 2
            rhs[3] = numerator_1 * numerator_2 / denominator_1
            return rhs

        return rhs_func

    def _get_triple_pendulum_rhs(self):
        l_1 = self.linkages[0].l
        l_2 = self.linkages[1].l
        l_3 = self.linkages[2].l
        m_11 = self.linkages[0].m_arm
        m_12 = self.linkages[0].m
        m_21 = self.linkages[1].m_arm
        m_22 = self.linkages[1].m
        m_31 = self.linkages[2].m_arm
        m_32 = self.linkages[2].m

        M_1 = (m_11 / 2 + m_12) / (m_11 + m_12)
        M_2 = (m_21 / 2 + m_22) / (m_21 + m_22)
        M_3 = (m_31 / 2 + m_32) / (m_31 + m_32)

        def rhs_func(t, y):
            rhs = np.zeros(len(self) * 2)
            theta_1 = y[0]
            theta_2 = y[1]
            theta_3 = y[2]
            omega_1 = y[3]
            omega_2 = y[4]
            omega_3 = y[5]
            g = self.g

            A11 = (l_1 ** 2) * ((4 * (m_11 + m_12) * ( M_1 ** 2) / 3) + 0.5*m_21 + m_22 + m_31 + m_32)
            A12 = l_1 * l_2 * np.cos(theta_1 - theta_2) * (m_21 / 2 + m_22 + m_31 + m_32) / 2
            A13 = l_1 * l_3 * M_3 * np.cos(theta_1 - theta_3)

            A21 = l_1 * l_2 * np.cos(theta_1 - theta_2) * (m_21 / 2 + m_22 + m_31 + m_32) / 2
            A22_term_1 = (m_21 + m_22) / 3
            A22_term_2 = l_1 ** 2 + l_2 ** 2 + M_2 ** 2 + + (M_2  * l_2) ** 2 / 3 + M_2 * l_1 * l_2 * np.cos(theta_1 - theta_2)
            A22_term_3 = l_2 ** 2 * (m_11 + m_21)
            A22 = A22_term_1 * A22_term_2 + A22_term_3
            A23 = (m_31 / 2 + m_32) * l_2 * l_3 * np.cos(theta_2 - theta_3) / 2

            A31 = (m_31 / 2 + m_32) * l_1 * l_3 * np.cos(theta_1 - theta_3) / 2
            A32 = (m_31 / 2 + m_32) * l_2 * l_3 * np.cos(theta_2 - theta_3) / 2

            A33_term_1 = (m_31 + m_32)
            A33_term_2 = l_1**2 + l_2**2 + 2*(M_3*l_3)**2 + l_1*l_2*np.cos(theta_1 - theta_2) + M_3*l_1*l_3*np.cos(theta_1 - theta_3) + \
                         M_3*l_2*l_3*np.cos(theta_2 - theta_3)
            A33 = A33_term_1*A33_term_2

            A = np.array([[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]])
            theta_dots = np.linalg.solve(A, np.array([[omega_1], [omega_2], [omega_3]]))
            theta_dot_1 = theta_dots[0]
            theta_dot_2 = theta_dots[1]
            theta_dot_3 = theta_dots[2]

            wdot_1_term_1 = -l_1*g*np.sin(theta_1)*(0.5*m_11 + m_12 + m_21 + m_22 + m_31 + m_32)
            wdot_1_term_2 = l_1*l_2*np.sin(theta_1 - theta_2)*((0.5*m_11 + m_22)*(theta_dot_1*theta_dot_2 + theta_dot_2 ** 2 / 3) +
                                                               (m_31 + m_32)*(theta_dot_1*theta_dot_2 + theta_dot_3 ** 2 / 3))
            wdot_1_term_3 = l_1*l_3*np.sin(theta_1 - theta_3)*(0.5*m_31 + m_32)*(theta_dot_1*theta_dot_2 + theta_dot_3 ** 2 / 3)
            wdot_1 = wdot_1_term_1 - 0.5*(wdot_1_term_2 + wdot_1_term_3)

            wdot_2_term_1 = -l_2*g*np.sin(theta_2)*(0.5*m_21 + m_22 + m_31 + m_32)
            wdot_2_term_2 = l_1*l_2*np.sin(theta_1 - theta_2)*((0.5*m_21 + m_22)*(theta_dot_1*theta_dot_2 + theta_dot_2 ** 2 / 3) +
                                                               (m_31 + m_32)*(theta_dot_1*theta_dot_3 + theta_dot_3 ** 2 / 3))
            wdot_2_term_3 = l_2*l_3*np.sin(theta_2 - theta_3)*(0.5*m_31 + m_32)*(theta_dot_2*theta_dot_3 + theta_dot_3 ** 2 / 3)
            wdot_2 = wdot_2_term_1 + 0.5*(wdot_2_term_2 - wdot_2_term_3)

            wdot_3_term_1 = -l_3*(0.5*m_31 + m_32)
            wdot_3_term_2 = g*np.sin(theta_3)
            wdot_3_term_3 = 0.5*(l_1*np.sin(theta_1 - theta_3)*(theta_dot_1*theta_dot_3 + theta_dot_3 ** 2 / 3) +
                                 l_2*np.sin(theta_2 - theta_3)*(theta_dot_2*theta_dot_3 + theta_dot_3 ** 2 / 3))
            wdot_3 = wdot_3_term_1 * (wdot_3_term_2 + wdot_3_term_3)

            rhs[0] = theta_dot_1
            rhs[1] = theta_dot_2
            rhs[2] = theta_dot_3
            rhs[3] = wdot_1
            rhs[4] = wdot_2
            rhs[5] = wdot_3
            return rhs
        return rhs_func

    def get_ode_rhs(self):
        if len(self) == 2:
            return self._get_double_pendulum_rhs()
        elif len(self) == 3:
            return self._get_triple_pendulum_rhs()
        else:
            raise ValueError(f'Unavailable number of linkages ({len(self)}). Must be 2 or 3')

    def solve(self, t_end, method='RK45'):
        y0 = np.array([linkage.theta for linkage in self.linkages] + [linkage.omega for linkage in self.linkages])
        t_bound = (0, t_end)
        ode_solution = integrate.solve_ivp(self.get_ode_rhs(), t_bound, y0, method=method, max_step=0.1)
        ode_solution.y_degrees = np.rad2deg(ode_solution.y)

        if len(self) == 2:
            columns = ['theta1', 'theta2', 'omega1', 'omega2']
        else:
            columns = ['theta1', 'theta2', 'theta3', 'omega1', 'omega2', 'omega3']
        df = pd.DataFrame(data=ode_solution.y.T, index=ode_solution.t, columns=columns)
        df['time'] = df.index
        columns_deg = [c + '_deg' for c in columns]
        df[columns_deg] = np.rad2deg(df[columns])
        # df['energy'] = self.calculate_energy(ode_solution)
        self.solution = ode_solution
        self.df = df
        return ode_solution, df

    def calculate_energy(self, solution):
        l1 = self.linkages[0].l
        l2 = self.linkages[1].l
        m1 = self.linkages[0].m
        m2 = self.linkages[1].m

        theta_1 = solution.y[0, :]
        theta_2 = solution.y[1, :]
        omega_1 = solution.y[2, :]
        omega_2 = solution.y[3, :]

        term1 = 0.5 * (l1**2*(m1 + m2)*omega_1**2 + l2**2*m2*omega_2**2 +
                       2*l1*l2*m2*np.cos(theta_1 - theta_2)*omega_1*omega_2)
        term2 = self.g*l1*(m1 + m2)*np.cos(theta_1)
        term3 = self.g*l2*m2*np.cos(theta_2)
        return term1 + term2 + term3

    def animate_solution(self, df):
        x1 = self.linkages[0].l * np.sin(df.theta1.values)
        y1 = -self.linkages[0].l * np.cos(df.theta1.values)

        x2 = self.linkages[1].l * np.sin(df.theta2.values) + x1
        y2 = -self.linkages[1].l * np.cos(df.theta2.values) + y1

        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False)
        ax.set_aspect('equal')
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=2)
        dt = df.time.diff().mean()
        def init():
            line.set_data([], [])
            return line

        def animate(i):
            thisx = [0, x1[i], x2[i]]
            thisy = [0, y1[i], y2[i]]

            line.set_data(thisx, thisy)
            return line

        ani = animation.FuncAnimation(fig, animate, range(1, len(df)),
                                      interval=dt * 1000, blit=True, init_func=init)
        return ani

    def plot_linkage_position(self, linkage_num, ax=None):
        if not ax:
            fig, ax = plt.subplots()

        ax.plot(self.df.time, self.df[f'theta{linkage_num}_deg'])
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Angular Position of Linkage {linkage_num}')
        ax.grid()
        return ax

    def plot_linkage_velocity(self, linkage_num, ax=None):
        if not ax:
            fig, ax = plt.subplots()

        ax.plot(self.df.time, self.df[f'omega{linkage_num}_deg'])
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Angular Velocity of Linkage {linkage_num}')
        ax.grid()
        return ax

    def plot_all_linkage_variables(self):
        fig, axs = plt.subplots(2, len(self))

        for i in range(len(self)):
            self.plot_linkage_position(i + 1, axs[0, i])
            self.plot_linkage_velocity(i + 1, axs[1, i])
        return fig, axs



if __name__ == '__main__':
    linkage_1 = Linkage(3, 2, np.pi / 4, 0)
    linkage_2 = Linkage(1, 1, np.pi / 4, 0)
    pendulum = Pendulum([linkage_1, linkage_2], 1)
    solution, df = pendulum.solve(100)

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(solution.t, solution.y_degrees[0, :])
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Angular Position of Linkage 1')
    axs[0, 0].grid()

    axs[0, 1].plot(solution.t, solution.y_degrees[1, :])
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Angular Position of Linkage 2')
    axs[0, 1].grid()

    axs[1, 0].plot(solution.t, solution.y_degrees[2, :])
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Angular Velocity of Linkage 1')
    axs[1, 0].grid()

    axs[1, 1].plot(solution.t, solution.y_degrees[3, :])
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Angular Velocity of Linkage 2')
    axs[1, 1].grid()

    energy = pendulum.calculate_energy(solution)
    _, ax = plt.subplots()
    ax.plot(solution.t, energy)
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.grid()

    _, ax = plt.subplots()
    ax.plot(solution.y_degrees[2, :], solution.y_degrees[0, :])
    ax.set_xlabel('Angular Velocity of Linkage 1')
    ax.set_ylabel('Angular Position of Linkage 1')
    ax.grid()

    _, ax = plt.subplots()
    ax.plot(solution.y_degrees[3, :], solution.y_degrees[1, :])
    ax.set_xlabel('Angular Velocity of Linkage 2')
    ax.set_ylabel('Angular Position of Linkage 2')
    ax.grid()

    _, ax = plt.subplots()
    ax.plot(solution.y_degrees[0, :], solution.y_degrees[1, :])
    ax.set_xlabel('Angular Position of Linkage 1')
    ax.set_ylabel('Angular Position of Linkage 2')
    ax.grid()
    ax.set_aspect('equal')

    _, ax = plt.subplots()
    ax.plot(solution.y_degrees[2, :], solution.y_degrees[3, :])
    ax.set_xlabel('Angular Velocity of Linkage 1')
    ax.set_ylabel('Angular Velocity of Linkage 2')
    ax.grid()
    ax.set_aspect('equal')


    # ani = pendulum.animate_solution(df)
    L = sum(linkage.l for linkage in pendulum.linkages)
    x1 = pendulum.linkages[0].l * np.sin(df.theta1.values)
    y1 = -pendulum.linkages[0].l * np.cos(df.theta1.values)

    x2 = pendulum.linkages[1].l * np.sin(df.theta2.values) + x1
    y2 = -pendulum.linkages[1].l * np.cos(df.theta2.values) + y1
    dt = df.time.diff().mean()
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    trace, = ax.plot([], [], ',-', lw=1)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    history_len = 500
    history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)


    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]

        if i == 0:
            history_x.clear()
            history_y.clear()

        history_x.appendleft(thisx[2])
        history_y.appendleft(thisy[2])

        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        time_text.set_text(time_template % (i * dt))
        return line, trace, time_text


    ani = animation.FuncAnimation(
        fig, animate, len(df), interval=dt * 100, blit=True)
    ani.save('pendulum.mp4')
    plt.show()





