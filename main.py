import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as integrate
import seaborn as sns


class Linkage:
    def __init__(self, mass, length, initial_angle, initial_angular_velocity):
        """
        Class representing a single linkage of a pendulum.
        :param mass: (float) Mass of the pendulum
        :param length: (float) Length of the pendulum
        :param initial_angle: (float) Initial angle of the pendulum from the downwards vertical, positive is counter-clockwise
        :param initial_angular_velocity: (float) Initial angular velocity  positive is counter-clockwise
        """
        self.m = mass
        self.l = length
        self.theta = initial_angle
        self.omega = initial_angular_velocity


class Pendulum:
    def __init__(self, linkage_list, g):
        self.linkages = linkage_list
        self.g = g

    def __len__(self):
        return len(self.linkages)

    def get_ode_rhs(self):
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
            numerator_1 = np.sin(theta1 - theta2) * (l1*np.cos(theta1 - theta2) * omega1 ** 2 + omega2 ** 2)
            denominator_1 = 2 * l1 * (1 + m1 - np.cos(theta1 - theta2) ** 2)

            numerator_2 = (1 + 2*m1)*np.sin(theta1) + np.sin(theta1 - 2*theta2)
            denominator_2 = l1*(1 + m1 - np.cos(theta1 - theta2) ** 2)

            rhs[2] = numerator_1 / denominator_1 - numerator_2 / denominator_2

            # Define omega_dot_2
            numerator_1 = np.sin(theta1 - theta2)
            numerator_2 = (1 + m1) * (np.cos(theta1) + l1 * omega1 ** 2) + np.cos(theta1 - theta2) * omega2 ** 2
            denominator_1 = 1 + m1 - np.cos(theta1 - theta2) ** 2
            rhs[3] = numerator_1 * numerator_2 / denominator_1
            return rhs
        return rhs_func

    def solve(self, t_end, method='RK45'):
        y0 = np.array([self.linkages[0].theta, self.linkages[1].theta, self.linkages[0].omega, self.linkages[1].omega])
        t_bound = (0, t_end)
        solution = integrate.solve_ivp(self.get_ode_rhs(), t_bound, y0, method=method, max_step=0.1)
        solution.y_degrees = np.rad2deg(solution.y)

        df = pd.DataFrame(data=solution.y.T, index=solution.t, columns=['theta1', 'theta2', 'omega1', 'omega2'])
        df['time'] = df.index
        df.theta1_deg = np.rad2deg(df.theta1)
        df.theta2_deg = np.rad2deg(df.theta2)
        df.omega1_deg = np.rad2deg(df.omega1)
        df.omega2_deg = np.rad2deg(df.omega2)
        df['energy'] = self.calculate_energy(solution)
        return solution, df

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



if __name__ == '__main__':
    linkage_1 = Linkage(3, 2, np.pi / 100, 0)
    linkage_2 = Linkage(1, 1, 0, 0)
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

    _, ax = plt.subplots()
    ax.plot(solution.y_degrees[2, :], solution.y_degrees[3, :])
    ax.set_xlabel('Angular Velocity of Linkage 1')
    ax.set_ylabel('Angular Velocity of Linkage 2')
    ax.grid()

    plt.show()




