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
    def __init__(self, linkage_list, g, t_end):
        self.linkages = linkage_list
        self.g = g
        self.t_end = t_end
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
        m_1 = self.linkages[0].m
        m_2 = self.linkages[1].m
        m_3 = self.linkages[2].m


        def rhs_func(t, y):
            rhs = np.zeros(len(self) * 2)
            theta_1 = y[0]
            theta_2 = y[1]
            theta_3 = y[2]
            omega_1 = y[3]
            omega_2 = y[4]
            omega_3 = y[5]

            omega = np.array([[omega_1], [omega_2], [omega_3]])
            g = self.g

            A11 = l_1 ** 2 * (m_1/3 + m_2 + m_3)
            A12 = (m_2/4 + m_3/2)*l_1*l_2*np.cos(theta_1 - theta_2)
            A13 = m_3*l_1*l_3 / 4 * np.cos(theta_1 - theta_3)

            A21 = A12
            A22 = m_2*(l_1**2/3 + l_2**2/3 + l_1*l_2/6*np.cos(theta_1 - theta_2)) + m_3*l_2**2
            A23 = m_3*l_2*l_3/4*np.cos(theta_2 - theta_3)

            A31 = A13
            A32 = A23
            A33 = m_3/3*(l_1**2 + l_2**2 + l_3**2 + l_1*l_2*np.cos(theta_1 - theta_2) + l_1*l_3/2*np.cos(theta_1-theta_3) + l_2*l_3/2*np.cos(theta_2-theta_3))

            A = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]]


            dAdt11 = 0
            dAdt12 = -(m_2/4 + m_3/2)*l_1*l_2*(omega_1 - omega_2)*np.sin(theta_1 - theta_2)
            dAdt13 = -m_3*l_1*l_3/4*(omega_1 - omega_3)*np.sin(theta_1 - theta_3)

            dAdt21 = dAdt12
            dAdt22 = -m_2*l_1*l_2/6*(omega_1 - omega_2)*np.sin(theta_1 - theta_2)
            dAdt23 = -m_3*l_2*l_3/4*(omega_2 - omega_3)*np.sin(theta_2 - theta_3)

            dAdt31 = dAdt13
            dAdt32 = dAdt23
            dAdt33 = -m_3/3*(l_1*l_2*(omega_1 - omega_2)*np.sin(theta_1 - theta_2) + l_1*l_3/2*(omega_1 - omega_3)*np.sin(theta_1 - theta_3)
                              + l_2*l_3/2*(omega_2 - omega_3)*np.sin(theta_2 - theta_3))

            dAdt = [[dAdt11, dAdt12, dAdt13], [dAdt21, dAdt22, dAdt23], [dAdt31, dAdt32, dAdt33]]

            dldth_1_term_1 = -l_1*l_2 / 4 * np.sin(theta_1 - theta_2)
            dldth_1_term_2 = m_2*(omega_2**2 / 3 + omega_1*omega_2) + 2*m_3*(omega_1*omega_2 + omega_3**2/3)
            dldth_1_term_3 = -m_3*l_1*l_3/4 * np.sin(theta_1 - theta_3)*(omega_1*omega_3 + omega_3 ** 2 /3)
            dldth_1_term_4 = -g*l_1*np.sin(theta_1)*(m_1/2 + m_2 + m_3)

            dldth_1 = dldth_1_term_1*dldth_1_term_2 + dldth_1_term_3 + dldth_1_term_4

            dldth_2_term_1 = l_1*l_2/4*np.sin(theta_1 - theta_2)
            dldth_2_term_2 = m_2*(omega_2 **2 / 3 + omega_1*omega_2) + 2*m_3*(omega_3 ** 2/3 + omega_1*omega_2)
            dldth_2_term_3 = -m_3*l_2*l_3/4*np.sin(theta_2 - theta_3)*(omega_2*omega_3 + omega_3 ** 2 /3)
            dldth_2_term_4 = -g*l_2*np.sin(theta_2)*(m_2/2 + m_3)

            dldth_2 = dldth_2_term_1*dldth_2_term_2 + dldth_2_term_3 + dldth_2_term_4

            dldth_3_term_1 = m_3/4 * (l_1*l_3*np.sin(theta_1 - theta_3)*(omega_1*omega_3 + omega_3 ** 2 /3))
            dldth_3_term_2 = m_3/4 * l_2*l_3*np.sin(theta_2 - theta_3)*(omega_2*omega_3 + omega_3 ** 2/3)
            dldth_3_term_3 = -g*m_3*l_3/2*np.sin(theta_3)

            dldth_3 = dldth_3_term_1 + dldth_3_term_2 + dldth_3_term_3
            dldth = np.array([[dldth_1], [dldth_2], [dldth_3]])

            omega_dot = np.linalg.solve(A, dldth - dAdt*omega)

            rhs[0] = omega_1
            rhs[1] = omega_2
            rhs[2] = omega_3
            rhs[3] = omega_dot[0, 0]
            rhs[4] = omega_dot[1, 0]
            rhs[5] = omega_dot[2, 0]
            return rhs
        return rhs_func

    def get_ode_rhs(self):
        if len(self) == 2:
            return self._get_double_pendulum_rhs()
        elif len(self) == 3:
            return self._get_triple_pendulum_rhs()
        else:
            raise ValueError(f'Unavailable number of linkages ({len(self)}). Must be 2 or 3')

    def solve(self, method='RK45'):
        y0 = np.array([linkage.theta for linkage in self.linkages] + [linkage.omega for linkage in self.linkages])
        t_bound = (0, self.t_end)
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
        l3 = self.linkages[2].l
        m1 = self.linkages[0].m
        m2 = self.linkages[1].m
        m3 = self.linkages[2].m

        theta_1 = solution.y[0, :]
        theta_2 = solution.y[1, :]
        theta_3 = solution.y[2, :]
        omega_1 = solution.y[3, :]
        omega_2 = solution.y[4, :]
        omega_3 = solution.y[5, :]

        g = self.g

        mass_1_term_1 = m1*(0.5*l1**2 * omega_1**2 / 3 - 0.5*g*l1*np.cos(theta_1))
        term1 = mass_1_term_1

        mass_2_term_1 = 0.5*m2*(l1**2*(omega_1**2 + omega_2**2/3) + l2**2*omega_2**2/3 + 0.5*l1*l2*np.cos(theta_1 - theta_2)*(omega_2**2/3 + omega_1*omega_2))
        mass_2_term_2 = -g*m2*(l1*(np.cos(theta_1) + 0.5*l2*np.cos(theta_2)))

        term2 = mass_2_term_1 + mass_2_term_2

        mass_3_term_1 = 0.5*m3*(l1**2*(omega_1**2 + omega_3**2/3) + l2**2*(omega_2**2 + omega_3**2/3) + l3**2*omega_3**2/3)
        mass_3_term_2 = 0.5*m3*(l1*l2*np.cos(theta_1 - theta_2)*(omega_1*omega_2 + omega_3**2/3) + 0.5*l1*l3*np.cos(theta_1 - theta_3)*(omega_1*omega_3 + omega_3**2/3) +
                                0.5*l2*l3*np.cos(theta_2 - theta_3)*(omega_2*omega_3 + omega_3**2/3))
        mass_3_term_3 = -g*m3*(l1*np.cos(theta_1) + l2*np.cos(theta_2) + 0.5*l3*np.cos(theta_3))
        term3 = mass_3_term_1 + mass_3_term_2 + mass_3_term_3
        return term1 + term2 + term3

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

    def plot_energy(self):
        self.df['energy'] = self.calculate_energy(self.solution)
        fig, ax = plt.subplots()

        ax.plot(self.df.time, self.df.energy)
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')
        ax.grid()
        return ax

    def plot_all_linkage_variables(self):
        fig, axs = plt.subplots(2, len(self))

        for i in range(len(self)):
            self.plot_linkage_position(i + 1, axs[0, i])
            self.plot_linkage_velocity(i + 1, axs[1, i])
        return fig, axs

class PendulumPlotter:
    def __init__(self, pendulum, ax):
        self.is_triple = len(pendulum) == 3
        self.pendulum = pendulum

        self.L = sum(linkage.l for linkage in pendulum.linkages)
        self.x1 = pendulum.linkages[0].l * np.sin(pendulum.df.theta1.values)
        self.y1 = -pendulum.linkages[0].l * np.cos(pendulum.df.theta1.values)

        self.x2 = pendulum.linkages[1].l * np.sin(pendulum.df.theta2.values) + self.x1
        self.y2 = -pendulum.linkages[1].l * np.cos(pendulum.df.theta2.values) + self.y1

        self.dt = pendulum.df.time.diff().mean()

        self.ax = ax
        self.fig = ax.get_figure()
        ax.set_xlim([-self.L, self.L])
        ax.set_ylim([-self.L, self.L])
        ax.set_aspect('equal')
        ax.grid()

        self.line, = ax.plot([], [], 'o-', lw=2)
        self.trace1 = ax.scatter([self.x1[0]], [self.y1[0]], cmap=plt.get_cmap('Greens'), alpha=0.5)
        self.trace2 = ax.scatter([self.x2[0]], [self.y2[0]], cmap=plt.get_cmap('Oranges'), alpha=0.5)

        self.time_template = 'time = %.1fs'
        self.time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        self.history_len = 500
        self.history_x_1, self.history_y_1 = deque(maxlen=self.history_len), deque(maxlen=self.history_len)
        self.history_x_2, self.history_y_2 = deque(maxlen=self.history_len), deque(maxlen=self.history_len)

        if self.is_triple:
            self.x3 = pendulum.linkages[2].l * np.sin(pendulum.df.theta3.values) + self.x2
            self.y3 = -pendulum.linkages[2].l * np.cos(pendulum.df.theta3.values) + self.y2
            self.trace3 = ax.scatter([self.x3[0]], [self.y3[0]], cmap=plt.get_cmap('Blues'), alpha=0.5)
            self.history_x_3, self.history_y_3 = deque(maxlen=self.history_len), deque(maxlen=self.history_len)


    def get_return_val(self):
        if self.is_triple:
            return self.line, self.trace1, self.trace2, self.trace3, self.time_text
        return self.line, self.trace1, self.trace2, self.time_text

def animate_solution(pendulums):
    """
    Animates physical position of one or more pendulums. Can pass either a single instance of a Pendulum or a list of pendulms.
    Additional pendulums will be animated on subplots
    :param pendulums: (list) List of pendulums to animate. Can be a single element list
    :return:
    """


    fig, axs = plt.subplots(len(pendulums))
    if len(pendulums) == 1:
        axs = [axs]
    pendulum_plotters = [PendulumPlotter(pendulum, ax) for pendulum, ax in zip(pendulums, axs)]
    def animate(i):
        for plotter in pendulum_plotters:
            thisx = [0, plotter.x1[i], plotter.x2[i], plotter.x3[i]]
            thisy = [0, plotter.y1[i], plotter.y2[i], plotter.y3[i]]

            if i == 0:
                plotter.history_x_1.clear()
                plotter.history_y_1.clear()
                plotter.history_x_2.clear()
                plotter.history_y_2.clear()
                if plotter.is_triple:
                    plotter.history_x_3.clear()
                    plotter.history_y_3.clear()

            plotter.history_x_1.appendleft(thisx[1])
            plotter.history_y_1.appendleft(thisy[1])
            plotter.history_x_2.appendleft(thisx[2])
            plotter.history_y_2.appendleft(thisy[2])

            plotter.line.set_data(thisx, thisy)
            plotter.trace1.set_offsets(np.c_[plotter.history_x_1, plotter.history_y_1])
            plotter.trace1.set_cmap('Greens')
            plotter.trace1.set_clim(0, 1)
            plotter.trace1.set_sizes(1.0 * np.ones(len(plotter.history_x_1)))
            plotter.trace1.set_array(np.linspace(1, 0, len(plotter.history_x_1)))

            plotter.trace2.set_offsets(np.c_[plotter.history_x_2, plotter.history_y_2])
            plotter.trace2.set_cmap('Oranges')
            plotter.trace2.set_clim(0, 1)
            plotter.trace2.set_sizes(1.0 * np.ones(len(plotter.history_x_2)))
            plotter.trace2.set_array(np.linspace(1, 0, len(plotter.history_x_2)))

            if plotter.is_triple:
                plotter.history_x_3.appendleft(thisx[3])
                plotter.history_y_3.appendleft(thisy[3])
                plotter.trace3.set_offsets(np.c_[plotter.history_x_3, plotter.history_y_3])
                plotter.trace3.set_cmap('Blues')
                plotter.trace3.set_clim(0, 1)
                plotter.trace3.set_sizes(1.0 * np.ones(len(plotter.history_x_3)))
                plotter.trace3.set_array(np.linspace(1, 0, len(plotter.history_x_3)))

            plotter.time_text.set_text(plotter.time_template % (i * plotter.dt))
        return (element for plotter in pendulum_plotters for element in plotter.get_return_val())

    ani = animation.FuncAnimation(
        fig, animate, len(pendulum_plotters[0].pendulum.df), interval=pendulum_plotters[0].dt * 500, blit=True)
    return ani

if __name__ == '__main__':
    linkage_1 = Linkage(1, 1, np.pi / 2, 0)
    linkage_2 = Linkage(1, 1, np.pi / 2, 0)
    linkage_3 = Linkage(1, 1, np.pi / 2, 0)
    pendulum = Pendulum([linkage_1, linkage_2, linkage_3], 1, 20)
    solution, df = pendulum.solve()

    pendulum.plot_all_linkage_variables()
    pendulum.plot_energy()
    ani = animate_solution([pendulum])
    plt.show()







