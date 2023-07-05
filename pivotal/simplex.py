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


def zero_out_cj(program, column, *, tolerance):
    row = program.pivots.get(column=column)
    cj = program.c[column]
    if is_zero(cj, tolerance):
        return
    program.M[0] -= cj*program.M[row]


def do_pivot(program, pivot):
    row, column = pivot
    program.M[row] /= program.M[row, column]

    for i in range(1, program.n_constraints+1):
        if i == row:
            continue
        program.M[i] -= program.M[i, column]*program.M[row]


def find_pivot(program, *, tolerance):
    for j in range(program.n_vars):
        if program.pivots.has(column=j):
            continue
        if program.c[j] < -tolerance:
            min_value = math.inf
            min_i = None
            for i in range(1, program.n_constraints + 1):
                if is_zero(program.M[i, j], tolerance):
                    continue
                frac = program.M[i, -1]/program.M[i, j]
                if program.M[i, j] > tolerance and (frac-min_value) < -tolerance:
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


def run_simplex(program, *, max_iterations=math.inf, tolerance=1e-6):
    iterations = 0
    while (pivot := find_pivot(program)) is not None:
        old_pivot = program.pivots.get(row=pivot[0])
        do_pivot(program, pivot)
        program.pivots.delete(column=old_pivot)
        program.pivots.set(row=pivot[0], column=pivot[1])
        zero_out_cj(program, pivot[1], tolerance=tolerance)

        iterations +=1
        if iterations > max_iterations:
            return

    if np.any(program.A[0] < -tolerance):
        raise Unbounded("The program is unbounded, try adding more constraints.")


def solve(program, *, max_iterations, tolerance):
    aux_program = aux_problem(program, tolerance=tolerance)
    run_simplex(aux_program, tolerance=tolerance)

    if not is_zero(aux_program.value, tolerance):
        raise Infeasible("The program is infeasible, try removing some constraints.")

    if any(j >= program.n_vars for j in aux_program.pivots.cr):
        zeros = is_zero(aux_program.A[:, :program.n_vars], tolerance)
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
                if is_zero(aux_program.M[row, candidate_col], tolerance):
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
        if not is_zero(new_prog.c[column], tolerance):
            zero_out_cj(new_prog, column, tolerance=tolerance)

    run_simplex(new_prog, max_iterations=max_iterations, tolerance=tolerance)
    return new_prog.value, get_solution(new_prog)


def aux_problem(program, *, tolerance):
    m = program.n_constraints
    A = np.copy(np.c_[program.A, np.eye(m)])
    b = np.copy(program.b)
    c = np.concatenate((np.zeros(program.n_vars), np.ones(m)))
    pivots = {program.n_vars+i: i+1 for i in range(m)}

    aux_program = LinearProgram(A, b, c, pivots)
    for j in range(program.n_vars, program.n_vars+m):
        zero_out_cj(aux_program, j, tolerance=tolerance)
    return aux_program


def is_zero(v, tol):
    return np.isclose(v, 0, rtol=0, atol=tol)


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
