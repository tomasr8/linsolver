import pytest
import numpy as np

from linsolver.simplex import algo, TOL


def test_program_1():
    A = np.array([
        [1, 3],
        [4, 1],
    ], dtype=float)

    b = np.array([6, 5], dtype=float)
    c = np.array([1, 2], dtype=float)

    value, X = algo(A, b, c)
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

    algo(A, b, c)

    value, X = algo(A, b, c)
    X_true = [1, 0, 7]
    assert value == pytest.approx(-300, TOL)
    for x, xt in zip(X, X_true):
        assert x == pytest.approx(xt, TOL)
