import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import os
import time
import math
import mpl_toolkits.mplot3d.axes3d as axes3d
from bifurcation import generate_list_of_pendulums, generate_list_of_linkages, solve_pendulum
from main import Linkage, Pendulum, animate_solution

def create_n_sphere(dimensions,N,shift):
    """
    :param dimensions: (int) no. dimensions of sphere
    :param N: (int) no. points
    :param shift: (float), this is the final radius of the sphere
    :return: () points
    """
    norm = np.random.normal
    normal_deviates = norm(size=(N, dimensions))

    radius = np.sqrt((normal_deviates ** 2).sum(axis=0))
    points = normal_deviates / radius
    normalised_points = []

    for point in points:
        normalised = point/np.linalg.norm(point)*shift
        normalised_points.append(normalised*shift)

    return(normalised_points)



def solve_pendulums(pendulums, num_processes=os.cpu_count() - 2):
    """ solves a list of pendulums - code stolen from bifurcation.generate_bifurcation_diagram
    :param num_processes: ~~
    :param pendulums: (list) List of Pendulum instances
    :return: == list of pendulums, list of dataframes
    """
    with multiprocessing.Pool(num_processes) as pool:
        outputs = list(pool.imap(solve_pendulum, pendulums))
    # Split up the outputs
    pendulums = [x[0] for x in outputs]
    dfs = [x[1] for x in outputs]

    return pendulums,dfs


def generate_shifted_ICs(CriticalPoints, shifts):
    """

    :param CriticalPoints:
    :param shifts:
    :return:
    """
    Shifted_ICs = []
    count = 1
    for pendulum in CriticalPoints:
        pendulums = []
        for shift in shifts:
            shifted = [(pend+shif) for pend, shif in zip(pendulum,shift)]
            pendulums.append(shifted)
        Shifted_ICs.append(pendulums)
    return(Shifted_ICs)


def generate_shifted_pendulums(Shifted_ICs,t_end,m=None,l=None):
    """

    :param Shifted_ICs: triple nested list: top layer = 8 fiducial groupings, next layer = 100 variations per fiducial, next layer = 6 ICs
    :param t_end:
    :return:
    """
    m = np.ones(len(Shifted_ICs[0]))
    l = np.ones(len(Shifted_ICs[0]))
    Pendulums = []

    for fiducial in Shifted_ICs:
        first_angles = [arm[0] for arm in fiducial]
        second_angles = [arm[1] for arm in fiducial]
        third_angles = [arm[2] for arm in fiducial]
        first_velocities = [arm[3] for arm in fiducial]
        second_velocities = [arm[4] for arm in fiducial]
        third_velocities = [arm[5] for arm in fiducial]

        first_linkages = generate_list_of_linkages(m,l,first_angles,first_velocities)
        second_linkages = generate_list_of_linkages(m,l,second_angles,second_velocities)
        third_linkages = generate_list_of_linkages(m,l,third_angles,third_velocities)

        shifted_pendulums = generate_list_of_pendulums(first_linkages,second_linkages,third_linkages,1,t_end)
        Pendulums.append(shifted_pendulums)

    return Pendulums


def generate_fiducial_links(CriticalPoints,omega, m, l):
    """
    :param first_fiducial_angles: the central initial conditions (critical points)
    :param second_fiducial_angles:
    :param third_fiducial_angles:
    :param omega:
    :return: list of pendulums
    """
    first_fiducial_angles = [angle[0] for angle in CriticalPoints]
    second_fiducial_angles = [angle[1] for angle in CriticalPoints]
    third_fiducial_angles = [angle[2] for angle in CriticalPoints]

    fiducial_linkages_1 = generate_list_of_linkages(m, l, first_fiducial_angles, omega)
    fiducial_linkages_2 = generate_list_of_linkages(m, l, second_fiducial_angles, omega)
    fiducial_linkages_3 = generate_list_of_linkages(m, l, third_fiducial_angles, omega)
    fiducial_linkages = fiducial_linkages_1, fiducial_linkages_2, fiducial_linkages_3
    return fiducial_linkages


def generate_fiducial_pendulums(fiducial_linkages, t_end):
    """Generate the fiducial (central) pendulums to which the 6 shifted ones are compared """
    fiducial_pendulums = generate_list_of_pendulums(fiducial_linkages[0], fiducial_linkages[1], fiducial_linkages[2], 1,
                                                    t_end)
    return (fiducial_pendulums)


def group(fiducials, Shifted_Pendulums):
    """groups pendulums with their fiducial pendulum for analysis"""
    df_groups = list(map(list, zip(fiducials, Shifted_Pendulums)))
    return df_groups


def distance(df1,df2,time):
    """calculates distance between two pendulum arms (in phase space?) at the final time.


    :param df1: (pandas DataFrame) first pandas series taken from a pendulum dataframe
    :param df2: (pandas DataFrame) second pandas series taken from a pendulum dataframe
    :param time: (float)
    :return: (int) distance
    """
    array1 = df1.iloc[time*10 + 1][0:6].array
    array2 = df2.iloc[time*10 + 1][0:6].array
    distance = np.linalg.norm(array1-array2)
    return(distance)


def calculate_lyapunov(Shifted_Pendulums,Fiducial_Pendulums,time_end,shift):
    """

    :param dfs_list: list of lists; nested list is of dataframes with similar IC's
    :param t_final: (int)
    :param shift: (float)
    :return:
    """
    exponents = []
    for pendulum in Fiducial_Pendulums:
        max_dist = 0
        index = Fiducial_Pendulums.index(pendulum)
        for shifted_pendulum in Shifted_Pendulums[index]:
            dist = distance(pendulum.df,shifted_pendulum.df,time_end)
            if dist>max_dist:
                max_dist = dist
        exponent = (1/t_end)*math.log2(max_dist/shift)      #note that the initial distance has been normalised to 1 in all cases
        exponents.append(exponent)
    return(exponents)

def lyapunov(CriticalPoints,n,t_end,dim,N,shift):
    """

    :param CriticalPoints:
    :param n:
    :param t_end:
    :param dim:
    :param N:
    :return:
    """

    m = np.ones(n)
    l = np.ones(n)
    omega = np.zeros(n)
    # Generate the 8 fiducial(central) pendulums stored in a list
    fiducial_linkages = generate_fiducial_links(CriticalPoints, omega, m, l)
    fiducial_pendulums = generate_fiducial_pendulums(fiducial_linkages, t_end)
    fiducial_pendulums, fiducial_dfs = solve_pendulums(fiducial_pendulums)

    # Create shifts by generating points on 6 dimensional unit sphere
    shifts = create_n_sphere(dim, N,shift)  # list of length 100, with inner lists of length 6

    Shifted_ICs = generate_shifted_ICs(CriticalPoints, shifts)

    Shifted_Pendulums = generate_shifted_pendulums(Shifted_ICs, t_end)
    Solved_Shifted_Pendulums = []
    for Pendulum_group in Shifted_Pendulums:
        t00 = time.time()
        print('solving pendulum group:', Shifted_Pendulums.index(Pendulum_group)+1)
        pendulums, dfs = solve_pendulums(Pendulum_group)
        Solved_Shifted_Pendulums.append(pendulums)
        print(f'took {time.time() - t00:.2f}s')

    times_list = [i for i in range(t_end)]

    Time_exponents = []
    for time_ in times_list:
        print('calculating exponent for time: ', time_)
        t0 = time.time()
        lyapunov_exponents = calculate_lyapunov(Solved_Shifted_Pendulums, fiducial_pendulums, time_,shift)
        Time_exponents.append(lyapunov_exponents)
        print(f'took {time.time() - t0:.2f}s')

    return(Time_exponents, times_list)


if __name__ == '__main__':
    #create parameters to construct the pendulums
    n = 8
    t_end = 200
    dim = 6
    N = 50
    shift = 0.000001
    CriticalPoints = [[0, 0, 0, 0, 0, 0], [0, 0, np.pi, 0, 0, 0], [0, np.pi, 0, 0, 0, 0], [0, np.pi, np.pi, 0, 0, 0],
                      [np.pi, 0, 0, 0, 0, 0], [np.pi, 0, np.pi, 0, 0, 0], [np.pi, np.pi, 0, 0, 0, 0],
                      [np.pi, np.pi, np.pi, 0, 0, 0]]

    Time_exponents, times_list= lyapunov(CriticalPoints,n,t_end,dim,N,shift)
    print(Time_exponents)

    exponent_1 = [exponents[0] for exponents in Time_exponents]
    exponent_2 = [exponents[1] for exponents in Time_exponents]
    exponent_3 = [exponents[2] for exponents in Time_exponents]
    exponent_4 = [exponents[3] for exponents in Time_exponents]
    exponent_5 = [exponents[4] for exponents in Time_exponents]
    exponent_6 = [exponents[5] for exponents in Time_exponents]
    exponent_7 = [exponents[6] for exponents in Time_exponents]
    exponent_8 = [exponents[7] for exponents in Time_exponents]

    plt.plot(times_list,exponent_1, label = 'E_1')
    plt.plot(times_list,exponent_2, label = 'E_2')
    plt.plot(times_list,exponent_3, label = 'E_3')
    plt.plot(times_list,exponent_4, label = 'E_4')
    plt.plot(times_list,exponent_5, label = 'E_5')
    plt.plot(times_list,exponent_6, label = 'E_6')
    plt.plot(times_list,exponent_7, label = 'E_7')
    plt.plot(times_list,exponent_8, label = 'E_8')
    # plt.title('Estimated Lyapunov Exponents')
    plt.xlabel('Time (s)')
    plt.ylabel('Exponent Value')
    plt.legend()
    plt.show()













#notes to self --> find a way to display the graphs of angle vs time to check if they're all starting in the correct position; if they are then can just ignore the really wacky plots/animations and carry on
