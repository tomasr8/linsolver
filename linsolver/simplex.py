import math
import numpy as np
from textwrap import indent
from dataclasses import dataclass


class LinearProgram:
    def __init__(self, A, b, c, pivots: dict[int, int]):
        self.M = np.block([
            [np.atleast_2d(c), np.atleast_2d(0)],
            [A, np.atleast_2d(b).T]
        ])
        self.n_vars = A.shape[1]
        self.n_constraints = A.shape[0]
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
        return -self.M[0, -1]

    def __str__(self):
        return f"{self.M}\n{self.pivots}"


def zero_out_cj(program, column):
    row = program.pivots[column]
    cj = program.c[column]
    if cj == 0:
        return
    program.M[0] -= cj*program.M[row]


def do_pivot(program, pivot):
    row, column = pivot
    program.M[row] /= program.M[row, column]

    for i in range(1, program.n_constraints+1):
        if i == row:
            continue
        program.M[i] -= program.M[i, column]*program.M[row]


def find_pivot(program):
    for j in range(program.n_vars):
        if j in program.pivots:
            continue
        # print("J", j)
        if program.c[j] < 0:
            # print("HERE")
            min_value = math.inf
            min_i = None
            for i in range(1, program.n_constraints + 1):
                frac = program.M[i, -1]/program.M[i, j]
                if program.M[i, j] > 0 and frac < min_value:
                    min_value = frac
                    min_i = i
            if min_i is not None:
                return min_i, j
    return None


def get_solution(program):
    n = program.n_vars
    solution = np.zeros(n, dtype=float)
    for j in range(n):
        if j in program.pivots:
            row = program.pivots[j]
            solution[j] = program.b[row-1]
    return solution


def run_simplex(program):
    print("in simplex")
    while (pivot := find_pivot(program)) is not None:
        print("[1st phase] pivot", pivot, 'old', get_column(program.pivots, pivot[0]))
        old_pivot = get_column(program.pivots, pivot[0])
        do_pivot(program, pivot)
        del program.pivots[old_pivot]
        program.pivots[pivot[1]] = pivot[0]
        print("after pivot")
        print(program.M)
        zero_out_cj(program, pivot[1])
        print("after zer out")
        print(program.M)

    if np.all(program.M[0, :-1] >= 0):
        return
    else:
        raise Exception("Unlimited")


def solve(program):
    aux_program = aux_problem(program)

    print("AUX")
    print(aux_program.M)
    print(aux_program.c)

    run_simplex(aux_program)


    print("MMM")
    print(aux_program.M)

    if aux_program.value > 0:
        print(aux_program.value)
        raise Exception("Infeasible")

    print("PIVOTS", aux_program.pivots)

    if any(j >= program.n_vars for j in aux_program.pivots):
        raise Exception("Degenerate solution")

    A = aux_program.A[:, :program.n_vars]
    b = np.copy(aux_program.b)
    c = np.copy(program.c)
    pivots = aux_program.pivots
    new_prog = LinearProgram(A, b, c, pivots)

    for column in pivots:
        if new_prog.c[column] != 0:
            zero_out_cj(new_prog, column)

    run_simplex(new_prog)
    print('[finished]', new_prog.value)
    print(new_prog.M)
    return new_prog.value, get_solution(new_prog)


def aux_problem(program):
    m = program.n_constraints
    A = np.copy(np.c_[program.A, np.eye(m)])
    b = np.copy(program.b)
    c = np.concatenate((np.zeros(program.n_vars), np.ones(m)))
    pivots = {program.n_vars+i: i+1 for i in range(m)}
    print("pivots", pivots)
    aux_program = LinearProgram(A, b, c, pivots)
    for j in range(program.n_vars, program.n_vars+m):
        print("zero out")
        print(aux_program.M)
        zero_out_cj(aux_program, j)
    print("final")
    print(aux_program.M)
    return aux_program


TOL = 1e-5


def equal(a, b, tol=TOL):
    return math.abs(a-b) <= tol


def canonicalize(type, A, b, c, constraint_types):
    if type == 'max':
        c *= -1

    # convert <= to >=
    sel = constraint_types == -1
    b[sel] *= -1
    A[sel] *= -1
    constraint_types[sel] = 1

    # convert >= to ==
    sel = constraint_types == 1
    n = np.sum(sel)

    if n > 0:
        print(A, np.zeros(A.shape[0], n))
        A = np.hstack((A, np.zeros((A.shape[0], n))))
        c = np.hstack((c, np.zeros(n)))
        A[sel, -n:] = -np.eye(n)
        constraint_types[sel] = 0

    # TODO: variable bounds
    # for j in range(A.shape[1]):
    #     AA = np.r_[AA, np.zeros(A.shape[1])]
    #     b = np.c_[b, 0]
    #     AA[-1, j] = 1
    #     AA[-1, -2] = -1
    #     AA[-1, -1] = 1
    #     c = np.c_[c, 0, 0]

    # convert -b to b
    sel = b < 0
    b[sel] *= -1
    A[sel] *= -1

    return A, b, c


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
