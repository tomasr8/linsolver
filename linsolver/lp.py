import itertools

import numpy as np


class Comparable:
    def __eq__(self, other):
        return Eq(self, other)

    def __ge__(self, other):
        return Gte(self, other)

    def __le__(self, other):
        return Lte(self, other)


class Variable(Comparable):
    id_iter = itertools.count()

    def __init__(self, name=None, coeff=1, lower_bound=0):
        if name is None:
            id = next(self.id_iter)
            self.name = f"_{id}"
        else:
            self.name = name
        self.coeff = coeff

    def __neg__(self):
        return Variable(name=self.name, coeff=-self.coeff)

    def __add__(self, other):
        match other:
            case Variable(name=name, coeff=coeff) if name == self.name:
                return Variable(name=name, coeff=self.coeff+coeff)
            case Variable() | int() | float():
                return Sum(self, other)
            case _:
                return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        return self.__add__(-other)

    __rsub__ = __sub__

    def __mul__(self, other):
        match other:
            case int() | float():
                return Variable(self.name, coeff=self.coeff * other)
            case _:
                raise TypeError("Can only multiply by a number")

    __rmul__ = __mul__

    def __repr__(self):
        if self.coeff == 1.0:
            return self.name
        elif self.coeff == -1.0:
            return f"-{self.name}"
        else:
            return f"{self.coeff}*{self.name}"


def simplify_sum(exprs):
    c = 0
    variables = {}
    for expr in exprs:
        match expr:
            case int() | float():
                c += expr
            case Variable(name=name):
                if name in variables:
                    variables[name] += expr
                else:
                    variables[name] = expr
            case _:
                raise Exception("cannot happen")
    return variables, c


def simplify(expr):
    match expr:
        case int() | float():
            return {}, expr
        case Variable():
            return {expr.name: expr}, 0
        case Sum(exprs=exprs):
            return simplify_sum(exprs)


class Sum(Comparable):
    def __init__(self, *exprs):
        self.exprs = exprs

    def __neg__(self):
        return Sum(*(-expr for expr in self.exprs))

    def __add__(self, other):
        match other:
            case Sum(exprs=exprs):
                return Sum(*self.exprs, *exprs)
            case Variable() | int() | float():
                return Sum(*self.exprs, other)
            case _:
                return NotImplemented

    def __sub__(self, other):
        return self.__add__(-other)

    def __radd__(self, other):
        match other:
            case Sum(exprs=exprs):
                return Sum(*exprs, *self.exprs)
            case Variable() | int() | float():
                return Sum(other, *self.exprs)
            case _:
                return NotImplemented

    def __repr__(self):
        # out = repr(self.exprs[0])
        # for expr in self.exprs[1:]:
        #     if 
        exprs = " + ".join(repr(x) for x in self.exprs)
        return f"{exprs}"


class Comparison:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def normalize(self):
        vars_left, c_left = simplify(self.left)
        vars_right, c_right = simplify(self.right)

        c = c_right - c_left
        variables = vars_left
        for name in vars_right:
            variables[name] = variables.get(name, 0) - vars_right[name]

        return variables, c


class Eq(Comparison):
    def __str__(self):
        return f"{self.left} = {self.right}"


class Gte(Comparison):
    def __str__(self):
        return f"{self.left} >= {self.right}"


class Lte(Comparison):
    def __str__(self):
        return f"{self.left} <= {self.right}"


_type = {
    Eq: 0,
    Gte: 1,
    Lte: -1,
}


class LinearProgram:
    MIN = 'min'
    MAX = 'max'

    def __init__(self, type, objective, constraints):
        self.type = type
        self.objective = objective
        self.constraints = constraints

    def as_matrix(self):
        variables = sorted(list(self._get_variables()))
        n_vars = len(variables)
        n_constraints = len(self.constraints)
        A = np.zeros((n_constraints, n_vars), dtype=float)
        b = np.zeros(n_constraints, dtype=float)
        c = np.zeros(n_vars, dtype=float)
        constraint_types = np.zeros(n_constraints, dtype=int)

        _vars, _ = simplify(self.objective)
        for name in _vars:
            i = variables.index(name)
            c[i] = _vars[name].coeff

        for row, constraint in enumerate(self.constraints):
            _vars, _b = simplify(constraint)
            for name in _vars:
                column = variables.index(name)
                A[row, column] = _vars[name].coeff
            b[row] = _b
            constraint_types[row] = _type[type(constraint)]

        return self.type, A, b, c, constraint_types

    def _get_variables(self):
        variables = set()
        _vars, _ = simplify(self.objective)
        variables += set(_vars.keys())
        for constraint in self.constraints:
            _vars, _ = simplify(constraint)
            variables += set(_vars.keys())
        return variables

    @classmethod
    def minimize(cls, objective):
        return cls(LinearProgram.MIN, objective, [])

    def such_that(self, *constraints):
        self.constraints = constraints
        return self

    def optimize(self):
        ...

    def __str__(self):
        constraints = "\n".join((f"  {constraint}" for constraint in self.constraints))
        return f"{self.type} {self.objective}\ns.t.\n{constraints}"


# X + (Y*2 - 4) + X = 23

X = Variable("X")
Y = Variable("Y")

# print(X)
# print(Y*2 - 4)
# print(X + (Y*2 - 4) + X)
# print(X + -(Y*2 - 4) + X)

# print(simplify(X + -(Y*2 - 4) + X))

# print(X >= 5)
# print(X == 5)
# print(X <= 5)
# print(5 <= X)
# print(5 == X + Y)
# print(5 >= X + Y)


# eq = ((X + (Y*2 - 4) + X) == 23)

# print(eq.left)
# print(eq.right)

# print("=====")
# print(eq.normalize())


program = LinearProgram.minimize(2*X + Y).such_that(X - Y >= 2, X + Y == 3)
print(program)
