import pytest
import numpy as np
from numpy.testing import assert_array_equal

from linsolver.simplex import solve, TOL, LinearProgram, canonicalize, zero_out_cj


def test_zero_out_cj_1():
    A = np.eye(2)
    b = np.ones(2)
    c = np.zeros(2)
    program = LinearProgram(A, b, c, {0: 1, 1: 2})
    M = np.copy(program.M)
    zero_out_cj(program, 0)
    assert_array_equal(M, program.M)
    zero_out_cj(program, 1)
    assert_array_equal(M, program.M)


def test_zero_out_cj_2():
    A = np.eye(2)
    b = np.ones(2)
    c = 2*np.ones(2)
    program = LinearProgram(A, b, c, {0: 1, 1: 2})
    zero_out_cj(program, 0)
    assert_array_equal(program.M, np.array([
        [0, 2, -2],
        [1, 0, 1],
        [0, 1, 1],
    ]))
    zero_out_cj(program, 1)
    assert_array_equal(program.M, np.array([
        [0, 0, -4],
        [1, 0, 1],
        [0, 1, 1],
    ]))


def test_canonicalize_1():
    A = np.eye(2)
    b = np.array([1, 2])
    c = np.array([3, 4])
    types = np.array([0, 0])

    Ac, bc, cc = canonicalize('min', A, b, c, types)
    assert_array_equal(A, Ac)
    assert_array_equal(b, bc)
    assert_array_equal(c, cc)


def test_canonicalize_2():
    A = np.array([[1, 1]])
    b = np.array([-2])
    c = np.array([3, 4])
    types = np.array([0])

    Ac, bc, cc = canonicalize('min', A, b, c, types)
    assert_array_equal(Ac, [
        [-1, -1]
    ])
    assert_array_equal(bc, [2])
    assert_array_equal(cc, [3, 4])


def test_canonicalize_3():
    A = np.array([[1, 1]])
    b = np.array([2])
    c = np.array([3, 4])
    types = np.array([1])

    Ac, bc, cc = canonicalize('min', A, b, c, types)
    assert_array_equal(Ac, [
        [1, 1, -1]
    ])
    assert_array_equal(bc, [2])
    assert_array_equal(cc, [3, 4, 0])


def test_canonicalize_4():
    A = np.array([[1, 1]])
    b = np.array([2])
    c = np.array([3, 4])
    types = np.array([-1])

    Ac, bc, cc = canonicalize('min', A, b, c, types)
    assert_array_equal(Ac, [
        [1, 1, 1]
    ])
    assert_array_equal(bc, [2])
    assert_array_equal(cc, [3, 4, 0])


def test_program_1():
    A = np.array([
        [1, 3],
        [4, 1],
    ], dtype=float)

    b = np.array([6, 5], dtype=float)
    c = np.array([1, 2], dtype=float)

    value, X = solve(LinearProgram(A, b, c, {}))
    X_true = [0.81818182, 1.72727273]
    assert value == pytest.approx(4.27272727, TOL)
    for x, xt in zip(X, X_true):
        assert x == pytest.approx(xt, TOL)


def test_program_2():
    A = np.array([
        [3, 2, 1],
        [1, 2, 2],
    ], dtype=float)

    b = np.array([10, 15], dtype=float)
    c = np.array([-20, -30, -40], dtype=float)

    value, X = solve(LinearProgram(A, b, c, {}))
    X_true = [1, 0, 7]
    assert value == pytest.approx(-300, TOL)
    for x, xt in zip(X, X_true):
        assert x == pytest.approx(xt, TOL)
