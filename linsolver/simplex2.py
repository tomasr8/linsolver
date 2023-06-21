import math
import numpy as np
from textwrap import indent
from dataclasses import dataclass


class LinearProgram:
    def __init__(self, M, pivots: dict[int, int]):
        self.M = M
        self.m = M.shape[0]
        self.n = M.shape[1]
        self.pivots = pivots

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

    def zero_out_cj(self, column):
        row = self.pivots[column]
        cj = self.c[column]
        if cj == 0:
            return
        self.M[0] -= cj*self.M[row]

    def do_pivot(self, pivot):
        row, column = pivot
        self.M[row] /= self.M[row, column]

        for i in range(1, self.m):
            if i == row:
                continue
            self.M[i] -= self.M[i, column]*self.M[row]

    def find_pivot(self):
        for j in range(self.n - 1):
            if j in self.pivots:
                continue
            if self.c[j] < 0:
                min_value = math.inf
                min_i = None
                for i in range(1, self.m):
                    frac = self.M[i, -1]/self.M[i, j]
                    if self.M[i, j] > 0 and frac < min_value:
                        min_value = frac
                        min_i = i
                if min_i is not None:
                    return min_i, j
        return None

    def get_solution(self):
        n = self.n - 1
        solution = np.zeros(n, dtype=float)
        for j in range(n):
            if j in self.pivots:
                row = self.pivots[j]
                solution[j] = self.b[row]
        return solution

    def run_simplex(self):
        while (pivot := self.find_pivot()) is not None:
            print("[1st phase] pivot", pivot, 'old', get_column(self.pivots, pivot[0]))
            old_pivot = get_column(self.pivots, pivot[0])
            self.do_pivot(pivot)
            del self.pivots[old_pivot]
            self.pivots[pivot[1]] = pivot[0]
            self.zero_out_cj(pivot[1])
            print(self.M)

        if np.all(self.M[0, :-1] >= 0):
            return -self.M[0, -1], self.M, self.pivots
        else:
            raise Exception("Unlimited")

    def solve(self):
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

    def aux_problem(self):
        A = np.c_[self.A, np.eye(self.m)]
        c = np.concatenate((np.zeros(self.n), np.ones(self.m)))

        M = np.zeros((A.shape[0]+1, A.shape[1]+1), dtype=float)
        M[1:, :-1] = A
        M[0, :-1] = c
        M[1:, -1] = self.b
        for j in range(self.n, A.shape[1]-1):
            self.zero_out_cj(M, self.pivots, j)
        return M, set(range(N, N+M))


TOL = 1e-5


def equal(a, b, tol=TOL):
    return math.abs(a-b) <= tol


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


def get_column(pivots, row):
    for column in pivots:
        if pivots[column] == row:
            return column

# def find_one(v):
#     for i in range(len(v)):
#         if v[i] == 1:
#             return i


# A = np.array([
#     [3, 2, 1],
#     [1, 2, 2],
# ], dtype=float)

# b = np.array([10, 15], dtype=float)
# c = np.array([-20, -30, -40], dtype=float)

# algo(A, b, c)


# A = np.array([
#     [1, 3],
#     [4, 1],
# ], dtype=float)

# b = np.array([6, 5], dtype=float)
# c = np.array([1, 2], dtype=float)

# algo(A, b, c)
