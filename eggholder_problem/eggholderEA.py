#!/usr/bin/env python3

""" eggholderEA.py: Implements an evolutionary algorithm for the eggholder
problem
"""

__author__ = 'Georgios Kouros'
__license__ = 'BSDv3'

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def egg_holder_ea() -> None:
    """ main function of the evolutionary algorithm solving the eggholder
    problem
    """
    alpha = 0.05  # mutation probability
    lambda_ = 100  # population and offspring size
    k = 3  # tournament selection
    int_max = 500  # Boundary of the domain, not intended to be changed

    # initialize population
    population = int_max * np.random.rand(lambda_, 2)

    plot_population(population, int_max)
    return

    for idx in range(0, 20):
        selected = selection(population, k)
        offspring = crossover(selected)
        joined_population = np.vstack((mutation(offspring, alpha, int_max),
                                       population))
        population = elimination(joined_population, lambda_)

        # show progress
        print(f'Iteration: {idx}, Mean fitness: {np.mean(objf(population))}')

        plot_population(population, int_max)


def objf(x: np.array) -> np.array:
    """ Computes the objective fucnction at the vector of (x,y) values

    Args:
        x (np.array): population

    Returns:
        np.array: 1D array containing the cost of the population samples
    """
    x = x.reshape((-1, 2))
    sas = np.sqrt(np.abs(x[:, 0] + x[:, 1]))
    sad = np.sqrt(np.abs(x[:, 0] - x[:, 1]))
    f_x = -x[:, 1] * np.sin(sas) - x[:, 0] * np.sin(sad)

    return f_x


def plot_population(population: np.array, int_max: int) -> None:
    """ Plot the population

    Args:
        population (np.array): The population to plot
        int_max (int): Boundary
    """
    x = np.linspace(0, int_max, 500).reshape((-1, 1))
    y = np.linspace(0, int_max, 500).reshape((-1, 1))
    F = -y.T * np.sin(np.sqrt(np.abs(x + y.T))) - x * np.sin(np.sqrt(np.abs(x - y.T)))

    Y, X = np.meshgrid(x, y)


    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, F,
                    rstride=8, cstride=8, shade=False, cmap="jet", linewidth=1, alpha=0.5)
    ax.scatter(population[:, 0],
               population[:, 1],
               objf(population)+1e-1, color='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Objective Funtion')
    ax.view_init(elev=80, azim=-135)

    plt.show()


def selection(population: np.array, k: int) -> np.array:
    """ Perform k-tournament selection to select pairs of parents """
    selected = zeros(2 * len(population), 2)

    for idx in range(0, 2 * lambda_):
        ri = randperm(lambda_, k)  # TODO replace
        _, mi = objf(population[ri, :]).min()
        selected[idx, :] = population[ri[mi], :]

    return selected


def crossover(selected: np.array) -> np.array:
    """ Perform crossover as in the slides

    Args:
        selected (np.array): The parents to produce the offsprings from

    Returns:
        np.array: The offsprings produced from the crossover operation
    """
    weights = 3 * np.random.rand(lambda_, 2) - 1
    offspring = np.zeros(lambda_, 2)

    for idx in range(1, len(offspring)):
        offspring[idx, 0] = min(int_max, max(0, selected[2 * idx - 1, 0]) + \
                weights[idx, 0] *\
                (selected[2 * idx, 0] - selected[2 * idx - 1, 0]))
        offspring[idx, 1] = min(int_max, max(0, selected[2 * idx - 1, 1]) + \
                weights[idx, 1] *\
                (selected[2 * idx, 1] - selected[2 * idx - 1, 1]))

    return offspring


def mutation(offspring: np.array, alpha: float, int_max: int) -> np.array:
    """ Perform mutation. adding a random gaussian perturbation

    Args:
        offspring (np.array): The produced offspring of the latest iteration
        alpha (float): Offset of offspring selection

    Returns:
        np.array: Mutated offspring
    """
    indices = np.random.rand(len(offspring), 1) <= alpha
    offspring[indices, :] = offspring[indices, :] + \
        10 * np.random.rand(len(indices, 2))
    offspring[indices, 0] = min(int_max, max(0, offspring[indices, 0]))
    offspring[indices, 1] = min(int_max, max(0, offspring[indices, 1]))

    return mutation


def elimination(joined_population: np.array, keep: int) -> np.array:
    """ Eliminate the unfit candidate solutions

    Args:
        joined_population (np.array): The population with the new offspring
        keep (int): The number of samples to keep from the population

    Returns:
        np.array: The samples that survived the elimination process
    """
    fvals = objf(joined_population)
    _, perm = fvals.sort()
    survivors = joined_population[perm[1:keep], :]

    return survivors


if __name__ == '__main__':
    print('Running the eggholder evolutionary algorithmic solver')
    egg_holder_ea()
