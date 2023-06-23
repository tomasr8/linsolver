import numpy as np
np.set_printoptions(precision=1, suppress=True)

# c = np.block([
#     [np.eye(1), np.atleast_2d(1)],
#     [np.eye(1), np.eye(1)]
# ])

# print(c)

from linsolver.lp import LinearProgram, Variable

# X = Variable("X")
# Y = Variable("Y")

# program = LinearProgram.minimize(2*X + Y).such_that(X - Y >= 2, X + Y == 3)
# print(program)
# print(program.optimize())


# b = Variable("b")
# m = Variable("m")
# z = Variable("z")

# program = LinearProgram.minimize(
#     25*b + 50*m + 80*z
# ).such_that(
#     2*b + m + z >= 8,
#     2*b + 6*m + z >= 16,
#     b + 3*m + 6*z >= 8,
# )

# print(program)
# print(program.optimize())


# Assignment problem
# C = np.array([
#     [1, 2, 3],
#     [2, 3, 1],
#     [10, 2, 1]
# ], dtype=float)

# X = [[Variable(f"x_{i}{j}") for j in range(3)] for i in range(3)]

# program = LinearProgram.minimize(
#     sum(sum(C[i, j] * X[i][j] for j in range(3)) for i in range(3))
# ).such_that(
#     sum(X[0][j] for j in range(3)) == 1,
#     sum(X[1][j] for j in range(3)) == 1,
#     sum(X[2][j] for j in range(3)) == 1,
#     sum(X[i][0] for i in range(3)) == 1,
#     sum(X[i][1] for i in range(3)) == 1,
#     sum(X[i][2] for i in range(3)) == 1,
# )

# print(program)
# print(program.optimize())


# C = np.array([
#     [1, 2, 3],
#     [2, 3, 1],
#     [10, 2, 1]
# ], dtype=float)

# X = [[Variable(f"x_{i}{j}") for j in range(2)] for i in range(2)]

# program = LinearProgram.minimize(
#     sum(sum(C[i, j] * X[i][j] for j in range(2)) for i in range(2))
# ).such_that(
#     sum(X[0][j] for j in range(2)) == 1,
#     sum(X[1][j] for j in range(2)) == 1,
#     sum(X[i][0] for i in range(2)) == 1,
#     sum(X[i][1] for i in range(2)) == 1,
# )

# print(program)
# print(program.optimize())


# X = [Variable(f"x{i+1}") for i in range(4)]

# program = LinearProgram.minimize(
#     X[0] + X[1] + 3*X[2]
# ).such_that(
#     X[0] + 5*X[1] + X[2] + X[3] == 7,
#     X[0] - X[1] + X[2] == 5,
#     0.5*X[0] - 2*X[1] + X[2] == 5,
# )

# print(program)
# print(program.optimize())


# X = [Variable(f"x{i+1}") for i in range(3)]

# program = LinearProgram.maximize(
#     X[0] + 2*X[1] - X[2]
# ).such_that(
#     2*X[0] - X[1] + X[2] == 12,
#     -X[0] + 2*X[1] + X[2] == 10,
#     X[0] + X[1] + 2*X[2] == 22,
# )

# print(program)
# print(program.optimize())
