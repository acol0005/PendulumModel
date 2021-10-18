import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import os
import time
import math
from bifurcation import generate_list_of_pendulums, generate_list_of_linkages, solve_pendulum
from main import Linkage, Pendulum, animate_solution


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


def generate_shifted_ICs(first_fiducial_angles, second_fiducial_angles, third_fiducial_angles, omega, shift):
    """ Generate the initial conditions to create the elipsoid around the central pendulum

    :param first_fiducial_angles: (list) list containing IC's of the first central arm
    :param second_fiducial_angles: (list) list containing IC's of the second central arm
    :param third_fiducial_angles: (list) list containing IC's of the third central arm
    :param omega: (list) starting intial velocities of each arm
    :param shift: (scalar) shift amount
    :return: (list) list of shifted angles, (list) list of shifted velocities
    """
    first_shifted_angles = [angle + shift for angle in first_fiducial_angles]
    second_shifted_angles = [angle + shift for angle in second_fiducial_angles]
    third_shifted_angles = [angle + shift for angle in third_fiducial_angles]
    omega1 = omega2 = omega3 = omega
    omega1[0] = omega2[1] = omega3[2] = shift
    shifted_angles = [first_shifted_angles, second_shifted_angles, third_shifted_angles]
    shifted_velocities = [omega1, omega2, omega3]
    return shifted_angles, shifted_velocities


def generate_shifted_links(fiducial_angles, shifted_angles, omega, shifted_velocities, m, l):
    """ takes central and shifted IC's and generates shifted linkages

    :param fiducial_angles:
    :param shifted_angles:
    :param omega:
    :param shifted_velocities:
    :param m:
    :param l:
    :return:
    """
    shifted_linkages_1 = generate_list_of_linkages(m, l, shifted_angles[0], omega)
    shifted_linkages_2 = generate_list_of_linkages(m, l, shifted_angles[1], omega)
    shifted_linkages_3 = generate_list_of_linkages(m, l, shifted_angles[2], omega)

    first_velocity_shifted = generate_list_of_linkages(m, l, fiducial_angles[0], shifted_velocities[0])
    second_velocity_shifted = generate_list_of_linkages(m, l, fiducial_angles[0], shifted_velocities[1])
    third_velocity_shifted = generate_list_of_linkages(m, l, fiducial_angles[0], shifted_velocities[2])

    shifted_first_links = [shifted_linkages_1, first_velocity_shifted]
    shifted_second_links = [shifted_linkages_2, second_velocity_shifted]
    shifted_third_links = [shifted_linkages_3, third_velocity_shifted]
    return shifted_first_links, shifted_second_links, shifted_third_links


def generate_shifted_pendulums(shifted_first_links, shifted_second_links, shifted_third_links, fiducial_linkages,
                               t_end):
    """takes shifted links and generates the pendulums
    :param t_end:
    :param shifted_first_links:
    :param shifted_second_links:
    :param shifted_third_links:
    :param fiducial_linkages:
    :return:
    """
    first_shifted_pendulums = generate_list_of_pendulums(shifted_first_links[0], fiducial_linkages[1],
                                                         fiducial_linkages[2],
                                                         1, t_end)
    second_shifted_pendulums = generate_list_of_pendulums(fiducial_linkages[0], shifted_second_links[0],
                                                          fiducial_linkages[2], 1, t_end)
    third_shifted_pendulums = generate_list_of_pendulums(fiducial_linkages[0], fiducial_linkages[1],
                                                         shifted_third_links[0], 1, t_end)
    fourth_shifted_pendulums = generate_list_of_pendulums(shifted_first_links[1], fiducial_linkages[1],
                                                          fiducial_linkages[2],
                                                          1, t_end)
    fifth_shifted_pendulums = generate_list_of_pendulums(fiducial_linkages[0], shifted_second_links[1],
                                                         fiducial_linkages[2], 1, t_end)
    sixth_shifted_pendulums = generate_list_of_pendulums(fiducial_linkages[0], fiducial_linkages[1],
                                                         shifted_third_links[1], 1, t_end)
    return first_shifted_pendulums, second_shifted_pendulums, third_shifted_pendulums, fourth_shifted_pendulums, fifth_shifted_pendulums, sixth_shifted_pendulums


def generate_fiducial_links(first_fiducial_angles, second_fiducial_angles, third_fiducial_angles, omega, m, l):
    """
    :param first_fiducial_angles: the central initial conditions (critical points)
    :param second_fiducial_angles:
    :param third_fiducial_angles:
    :param omega:
    :return: list of pendulums
    """
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


def group(fiducials, first_shifted, second_shifted,third_shifted,fourth_shifted,fifth_shifted,sixth_shifted):
    """groups pendulums with their fiducial pendulum for analysis"""
    df_groups = list(map(list, zip(fiducials, first_shifted, second_shifted, third_shifted, fourth_shifted, fifth_shifted, sixth_shifted)))
    return df_groups

def principle_distance(df1,df2):
    """calculates distance between two pendulum arms (in phase space?) at the final time.


    :param df1: (pandas DataFrame) first pandas series taken from a pendulum dataframe
    :param df2: (pandas DataFrame) second pandas series taken from a pendulum dataframe
    :return: (float) distance
    """
    distances = []
    for i in range(6):
        distances.append(abs(df1.iloc[len(df1)-1][i] - df2.iloc[len(df2)-1][i]))
    dist = max(distances)
    # print('distance is: ', dist)
    return dist


def calculate_lyapunov(dfs_list,t_final,shift):
    """

    :param dfs_list: list of lists; nested list is of dataframes with similar IC's
    :param t_final: (int)
    :param shift: (float)
    :return:
    """
    m = len(dfs_list)
    exponents = []
    for df in dfs_list[1:m+1]:
        numerator = principle_distance(df,dfs_list[0])
        if numerator != 0.0:
            exponent = (1/t_final)*math.log2(numerator/shift)
            exponents.append(exponent)
        else:
            exponents.append('zero distance')
    return(exponents)


if __name__ == '__main__':
    #create parameters to construct the pendulums
    n = 8
    t_end = 2000
    m = np.ones(n)
    l = np.ones(n)
    omega = np.zeros(n)
    shift = 0.01 #The shift in angular velocity applied to each pendulum arm

    CriticalPoints = [[0 ,0, 0], [np.pi, 0, 0], [0, np.pi, 0], [0, 0, np.pi], [np.pi, np.pi, 0],[np.pi, 0, np.pi],
                      [0, np.pi, np.pi], [0, np.pi, np.pi]]


    first_fiducial_angles = [item[0] for item in CriticalPoints]
    second_fiducial_angles = [item[1] for item in CriticalPoints]
    third_fiducial_angles = [item[2] for item in CriticalPoints]
    fiducial_angles = [first_fiducial_angles, second_fiducial_angles,third_fiducial_angles]

    #Generate the 8 fiducial(central) pendulums stored in a list
    fiducial_linkages = generate_fiducial_links(first_fiducial_angles, second_fiducial_angles, third_fiducial_angles, omega, m, l)
    fiducial_pendulums = generate_fiducial_pendulums(fiducial_linkages, t_end)


    shifted_angles, shifted_velocities = generate_shifted_ICs(first_fiducial_angles,second_fiducial_angles,third_fiducial_angles,omega,shift)
    shifted_first_links, shifted_second_links, shifted_third_links = generate_shifted_links(fiducial_angles, shifted_angles, omega, shifted_velocities, m, l)
    first_shifted_pendulums, second_shifted_pendulums, third_shifted_pendulums, fourth_shifted_pendulums, fifth_shifted_pendulums, sixth_shifted_pendulums = generate_shifted_pendulums(shifted_first_links, shifted_second_links, shifted_third_links, fiducial_linkages,
                               t_end)

    fiducial_pendulums, dfs = solve_pendulums(fiducial_pendulums)
    first_shifted_pendulums, first_dfs = solve_pendulums(first_shifted_pendulums)
    second_shifted_pendulums, second_dfs = solve_pendulums(second_shifted_pendulums)
    third_shifted_pendulums, third_dfs = solve_pendulums(third_shifted_pendulums)
    fourth_shifted_pendulums, fourth_dfs = solve_pendulums(fourth_shifted_pendulums)
    fifth_shifted_pendulums, fifth_dfs = solve_pendulums(fifth_shifted_pendulums)
    sixth_shifted_pendulums, sixth_dfs = solve_pendulums(sixth_shifted_pendulums)

    groups_dfs = group(dfs, first_dfs, second_dfs, third_dfs, fourth_dfs, fifth_dfs, sixth_dfs)
    groups_pendulums = group(fiducial_pendulums,first_shifted_pendulums,second_shifted_pendulums,third_shifted_pendulums,fourth_shifted_pendulums,fifth_shifted_pendulums,sixth_shifted_pendulums)
    #groups is a list of lists


    # for pendulum in groups_pendulums[6]:
    #     pendulum.plot_all_linkage_variables()
    #     plt.show()

    #
    for group in groups_dfs:
        exponents = calculate_lyapunov(group,t_end,shift)
        print(exponents)


#notes to self --> find a way to display the graphs of angle vs time to check if they're all starting in the correct position; if they are then can just ignore the really wacky plots/animations and carry on
