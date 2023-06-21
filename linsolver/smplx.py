import math
import numpy as np
from textwrap import indent


class LinearProgram:
    def __init__(self, M):
        self.M = M

    @property
    def A(self):
        return self.M[1:, :-1]

    @property
    def b(self):
        return self.M[1:, -1]

    @property
    def c(self):
        return self.M[0, :-1]

    @property
    def value(self):
        return self.M[0, -1]


TOL = 1e-5


def equal(a, b, tol=TOL):
    return math.abs(a-b) <= tol


def aux_problem(A, b, c):
    M, N = A.shape
    A = np.c_[A, np.eye(M)]
    c = np.concatenate((np.zeros(N), np.ones(M)))

    AA = np.zeros((M+1, M+N+1))
    AA[1:, :-1] = A
    AA[0, :-1] = c
    AA[1:, -1] = b
    for i in range(M):
        zero_out_cj(AA, (i+1, N+i))
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
    # print("  div")
    # print(indent(str(M), "  "))
    # print("==")

    for i in range(len(M)):
        if i == pi:
            continue
        M[i] -= M[i, pj]*M[pi]
        # print(indent(str(M), "  "))


def find_pivot(M, J):
    for j in range(len(M[0]) - 1):
        if j in J:
            continue
        if M[0, j] < 0:
            m = math.inf
            mi = None
            for i in range(1, len(M)):
                if M[i, j] > 0 and M[i, -1]/M[i, j] < m:
                    m = M[i, -1]/M[i, j]
                    mi = i
            if mi is not None:
                return mi, j
    return None


def get_old(M, J, pivot):
    for j in J:
        if M[pivot[0], j] == 1:
            return j


def basic_algo(M, J, it=100):
    while (pivot := find_pivot(M, J)) is not None:
        print("[1st phase] pivot", pivot, 'old', get_old(M, J, pivot))
        old_pivot = get_old(M, J, pivot)
        do_pivot(M, pivot)
        J.remove(old_pivot)
        J.add(pivot[1])
        zero_out_cj(M, pivot)

        print(M)
        it -= 1
        if it == 0:
            raise Exception("max iters")

    if np.all(M[0, :-1] >= 0):
        return -M[0, -1], M, J
    else:
        return "Unlimited", None, None


def algo(A, b, c):
    M, J = aux_problem(A, b, c)
    print("aux problem")
    print(M)
    print(J)
    value, M, J = basic_algo(M, J)

    print("1st phase", value)
    print(M)
    print(J)


    if value > 0:
        print(value)
        return "Infeasible"

    m, n = A.shape
    if any(j >= n for j in J):
        raise Exception("Degenerate solution")

    MM = np.zeros((A.shape[0]+1, A.shape[1]+1), dtype=float)
    MM[0, :-1] = c
    MM[:, -1] = M[:, -1]
    MM[1:, :-1] = M[1:, :A.shape[1]]


    for j in J:
        if MM[0, j] != 0:
            i = find_one(MM[1:, j])
            zero_out_cj(MM, (i+1, j))
    print("[2nd phase]")
    print(MM)
    # raise Exception("starting 2nd phase")
    value, M, J = basic_algo(MM, J, it=4)
    print('[finished]', value)
    print(M)
    return value, get_solution(M, J)


def is_base(v):
    return (v == 1).sum() == 1 and (v == 0).sum() == len(v) - 1


def find_one(v):
    for i in range(len(v)):
        if v[i] == 1:
            return i


def get_solution(M, J):
    solution = []
    for j in range(M.shape[1] - 1):
        if j not in J:
            solution.append(0)
        else:
            i = find_one(M[1:, j])
            solution.append(M[i+1, -1])
    return solution


# A = np.array([
#     [3, 2, 1],
#     [1, 2, 2],
# ], dtype=float)

# b = np.array([10, 15], dtype=float)
# c = np.array([-20, -30, -40], dtype=float)

# algo(A, b, c)


A = np.array([
    [1, 3],
    [4, 1],
], dtype=float)

b = np.array([6, 5], dtype=float)
c = np.array([1, 2], dtype=float)

algo(A, b, c)
