import math
import numpy as np


def canonicalize(A, b, c, t, m):
    if m == 'max':
        c *= -1

    AA = np.copy(A)

    for i in range(len(A)):
        if t[i] == 'eq' or t[i] == 'gt':
            continue
        if t[i] == 'lt':
            A[i] *= -1
            b[i] *= -1
            t[i] = 'gt'

    for j in range(len(c)):
        AA = np.r_[AA, np.zeros(len(c))]
        AA[-1, j] = 1
        AA[-1, -2] = -1
        AA[-1, -1] = 1
        c = np.c_[c, 0, 0]

    for i in range(len(A)):
        if t[i] == 'eq':
            continue
        if t[i] == 'gt':
            AA = np.c_[AA, np.zeros(len(AA))]
            AA[i, -1] = -1
            c = np.c_[c, 0]

    return AA, b, c, t, 'min'


# class LinearProgram:
#     pass


def zero_out_cj(M, pivot):
    i, j = pivot
    if M[0, j] == 0:
        return
    
    f = -M[0, j]
    for jj in range(len(M[0])):
        M[0, jj] += f*M[i, jj]


def do_pivot(M, pivot):
    pi, pj = pivot
    M[pi] /= M[pi, pj]

    for i in range(len(M)):
        if i == pi:
            continue
        M[i] -= M[i, pj]*M[pi]


def find_pivot(M, J):
    for j in range(len(M[0]-1)):
        if j in J:
            continue
        if M[0, j] < 0:
            m = math.inf
            mi = None
            for i in range(1, len(M)):
                if M[i, -1]/M[i, j] < m:
                    m = M[i, -1]/M[i, j]
                    mi = i
            if mi is not None:
                return mi, j
    return None


def get_old(M, J, pivot):
    for j in J:
        if M[pivot[0], j] == 1:
            return j


def basic_algo(M, J):
    while (pivot := find_pivot(M, J)) is not None:
        do_pivot(M, pivot)
        J.remove(get_old(M, J, pivot))
        J.add(pivot)
        zero_out_cj(M, pivot)

    if np.all(M[0, :-1] >= 0):
        return -M[0, -1]
    else:
        return "Unlimited"
