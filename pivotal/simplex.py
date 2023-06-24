import math

import numpy as np

from pivotal.errors import Infeasible, Unbounded


class Pivots:
    def __init__(self, cr):
        self.cr = cr
        self.rc = {cr[c]: c for c in cr}

    def get(self, *, row=None, column=None):
        if row is not None:
            return self.rc[row]
        else:
            return self.cr[column]

    def set(self, *, row, column):
        self.rc[row] = column
        self.cr[column] = row

    def has(self, *, row=None, column=None):
        if row is not None:
            return row in self.rc
        else:
            return column in self.cr

    def delete(self, *, row=None, column=None):
        if row is not None:
            column = self.rc[row]
            del self.rc[row]
            del self.cr[column]
        else:
            row = self.cr[column]
            del self.rc[row]
            del self.cr[column]


class LinearProgram:
    def __init__(self, A, b, c, pivots: dict[int, int] | Pivots):
        self.M = np.block([
            [np.atleast_2d(c), np.atleast_2d(0)],
            [A, np.atleast_2d(b).T]
        ])
        self.pivots = pivots if isinstance(pivots, Pivots) else Pivots(pivots)

    @property
    def n_vars(self):
        return self.A.shape[1]

    @property
    def n_constraints(self):
        return self.A.shape[0]

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
    row = program.pivots.get(column=column)
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
        if program.pivots.has(column=j):
            continue
        if program.c[j] < 0:
            min_value = math.inf
            min_i = None
            for i in range(1, program.n_constraints + 1):
                if program.M[i, j] == 0:
                    continue
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
        if program.pivots.has(column=j):
            row = program.pivots.get(column=j)
            solution[j] = program.b[row-1]
    return solution


def run_simplex(program):
    while (pivot := find_pivot(program)) is not None:
        old_pivot = program.pivots.get(row=pivot[0])
        do_pivot(program, pivot)
        program.pivots.delete(column=old_pivot)
        program.pivots.set(row=pivot[0], column=pivot[1])
        zero_out_cj(program, pivot[1])

    if np.all(program.M[0, :-1] >= 0):
        return
    else:
        raise Unbounded("The program is unbounded, try adding more constraints.")


def solve(program):
    aux_program = aux_problem(program)
    run_simplex(aux_program)

    if not is_zero(aux_program.value):
        raise Infeasible("The program is infeasible, try removing some constraints.")

    if any(j >= program.n_vars for j in aux_program.pivots.cr):
        zeros = np.isclose(aux_program.A[:, :program.n_vars], 0)
        sel = np.all(zeros, axis=1)
        sel = np.concatenate(([False], sel))

        if np.any(sel):
            aux_program.M = aux_program.M[~sel]
            for i, s in enumerate(sel):
                if s:
                    col = aux_program.pivots.get(row=i)
                    aux_program.pivots.delete(column=col)

        pivots = []
        for col in aux_program.pivots.cr:
            if col >= program.n_vars:
                pivots.append((aux_program.pivots.get(column=col), col))

        for p in pivots:
            row, col = p
            for candidate_col in range(program.n_vars):
                if candidate_col in aux_program.pivots.cr:
                    continue
                if is_zero(aux_program.M[row, candidate_col]):
                    continue
                do_pivot(aux_program, (row, candidate_col))
                aux_program.pivots.delete(column=col)
                aux_program.pivots.set(row=row, column=candidate_col)
                break

    A = aux_program.A[:, :program.n_vars]
    b = np.copy(aux_program.b)
    c = np.copy(program.c)
    pivots = aux_program.pivots
    new_prog = LinearProgram(A, b, c, pivots)

    for column in pivots.cr:
        if new_prog.c[column] != 0:
            zero_out_cj(new_prog, column)

    run_simplex(new_prog)
    return new_prog.value, get_solution(new_prog)


def aux_problem(program):
    m = program.n_constraints
    A = np.copy(np.c_[program.A, np.eye(m)])
    b = np.copy(program.b)
    c = np.concatenate((np.zeros(program.n_vars), np.ones(m)))
    pivots = {program.n_vars+i: i+1 for i in range(m)}

    aux_program = LinearProgram(A, b, c, pivots)
    for j in range(program.n_vars, program.n_vars+m):
        zero_out_cj(aux_program, j)
    return aux_program


TOL = 1e-5


def equal(a, b, tol=TOL):
    return abs(a-b) <= tol


def is_zero(v):
    return equal(v, 0)


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
