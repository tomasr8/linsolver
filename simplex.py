import math
import numpy as np


def aux_problem(A, b, c):
    M, N = A.shape
    A = np.c_[A, np.eye(M)]
    c = np.c_[np.zeros(N), np.ones(M)]

    AA = np.zeros((M+1, M+N+1))
    AA[:-1, :-1] = A
    AA[0, :] = c
    AA[1:, -1] = b
    for i in range(M):
        zero_out_cj(M, (N+i, i+1))
    return AA, set(range(N, N+M))


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
        b = np.c_[b, 0]
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
            b = np.c_[b, 0]

    for i in range(len(b)):
        if b < 0:
            b *= -1
            AA[i] *= -1

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
        return -M[0, -1], M, J
    else:
        return "Unlimited", None, None


def algo(A, b, c):
    M, J = aux_problem(A, b, c)
    value, M, J = basic_algo(M, J)

    if value > 0:
        print(value)
        return "Infeasible"

    m, n = A.shape
    if any(j >= n for j in J):
        raise Exception("Degenerate solution")


    M[0, :-1] = c
    for j in range(J):
        if M[0, j] != 0:
            i = find_one(M[1:, j])
            zero_out_cj(M, (i+1, j))
    value, M, J = basic_algo(M, J)
    return value, M, J


def is_base(v):
    return (v == 1).sum() == 1 and (v == 0).sum() == len(v) - 1

def find_one(v):
    for i in range(len(v)):
        if v[i] == 1:
            return i
