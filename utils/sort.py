import numpy as np
import random


def Dominates(x, y):
    """Check if x dominates y.
    :param x: a sample
    :type x: array
    :param y: a sample
    :type y: array
    """
    return np.all(x <= y) & np.any(x < y)


def NonDominatedSorting(pop):
    """Perform non-dominated sorting.
    :param pop: the current population
    :type pop: array
    """
    _, npop = pop.shape
    rank = np.zeros(npop)
    dominatedCount = np.zeros(npop)
    dominatedSet = [[] for i in range(npop)]
    F = [[]]
    for i in range(npop):
        for j in range(i + 1, npop):
            p = pop[:, i]
            q = pop[:, j]
            if Dominates(p, q):
                dominatedSet[i].append(j)
                dominatedCount[j] += 1
            if Dominates(q, p):
                dominatedSet[j].append(i)
                dominatedCount[i] += 1
        if dominatedCount[i] == 0:
            rank[i] = 1
            F[0].append(i)
    k = 0
    while (True):
        Q = []
        for i in F[k]:
            p = pop[:, i]
            for j in dominatedSet[i]:
                dominatedCount[j] -= 1
                if dominatedCount[j] == 0:
                    Q.append(j)
                    rank[j] = k + 1
        if len(Q) == 0:
            break
        F.append(Q)
        k += 1
    return F


def CARS_NSGA(target, objs, N):
    """pNSGA-III (CARS-NSGA).
    :param target: the first objective, e.g. accuracy
    :type target: array
    :param objs: the other objective, e.g. FLOPs, number of parameteres
    :type objs: array
    :param N: number of population
    :type N: int
    :return: The selected samples
    :rtype: array
    """
    selected = np.zeros(target.shape[0])
    Fs = []
    for obj in objs:
        Fs.append(NonDominatedSorting(np.vstack((1 / (target + 1e-10), obj))))
        Fs.append(NonDominatedSorting(
            np.vstack((1 / (target + 1e-10), 1 / (obj + 1e-10)))))
    stage = 0
    while (np.sum(selected) < N):
        current_front = []
        for i in range(len(Fs)):
            if len(Fs[i]) > stage:
                current_front.append(Fs[i][stage])
        current_front = [np.array(c) for c in current_front]
        current_front = np.hstack(current_front)
        current_front = list(set(current_front))
        if np.sum(selected) + len(current_front) <= N:
            for i in current_front:
                selected[i] = 1
        else:
            not_selected_indices = np.arange(len(selected))[selected == 0]
            crt_front = [
                index for index in current_front if index in not_selected_indices]
            num_to_select = N - np.sum(selected).astype(np.int32)
            current_front = crt_front if len(
                crt_front) <= num_to_select else random.sample(crt_front, num_to_select)
            for i in current_front:
                selected[i] = 1
        stage += 1
    return np.where(selected == 1)[0]
