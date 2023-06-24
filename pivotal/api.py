import itertools

import numpy as np

from pivotal.simplex import LinearProgram as LP
from pivotal.simplex import canonicalize, solve


class Expression:
    pass


class Abs(Expression):
    def __init__(self, arg, sign=1):
        self.arg = arg
        self.sign = sign

    def __abs__(self):
        return Abs(self.arg)

    def __neg__(self):
        return Abs(self.arg, -self.sign)

    def __add__(self, other):
        match other:
            case Abs():
                return Sum(self, other)
            case Sum(elts=elts):
                return Sum(self, *elts)
            case int() | float():
                if other == 0:
                    return self
                return Sum(self, other)
            case Variable() | int() | float():
                return Sum(self, other)
            case _:
                return NotImplemented

    def __sub__(self, other):
        return self.__add__(-other)

    def __radd__(self, other):
        match other:
            case int() | float():
                if other == 0:
                    return self
                return Sum(other, self)
            case _:
                return NotImplemented

    def __rsub__(self, other):
        return -self.__radd__(other)

    def __mul__(self, other):
        match other:
            case int() | float():
                if other == 1:
                    return self
                elif other == 0:
                    return 0
                elif other > 0:
                    return Abs(other*self.arg)
                else:
                    return Abs(abs(other)*self.arg, sign=-1)
            case _:
                return NotImplemented

    __rmul__ = __mul__

    def __str__(self):
        sign = "-" if self.sign == -1 else ""
        return f"{sign}|{self.arg}|"


class Comparable:
    def __eq__(self, other):
        return Eq(self, other)

    def __ge__(self, other):
        return Gte(self, other)

    def __le__(self, other):
        return Lte(self, other)


class Variable(Expression, Comparable):
    id_iter = itertools.count()

    def __init__(self, name: str | None = None, coeff=1.0, lb=0, ub=0):
        if name is None:
            id = next(self.id_iter)
            self.name = f"_{id}"
        # TODO:
        # elif name.startswith("_"):
            # raise TypeError("Variables starting with an underscore are reserved for slack"
                            # " and auxiliary variables used by the solver.")
        else:
            self.name = name
        self.coeff = coeff

    def __abs__(self):
        return Abs(self)

    def __neg__(self):
        return Variable(name=self.name, coeff=-self.coeff)

    def __add__(self, other):
        match other:
            case Variable(name=name, coeff=coeff) if name == self.name:
                if self.coeff+coeff == 0:
                    return 0
                return Variable(name=name, coeff=self.coeff+coeff)
            case int() | float() if other == 0:
                return self
            case Variable() | Abs() | int() | float():
                return Sum(self, other)
            case Sum(elts=elts):
                return Sum(self, *elts)
            case _:
                return NotImplemented

    def __radd__(self, other):
        match other:
            case int() | float():
                if other == 0:
                    return self
                return Sum(other, self)
            case _:
                return NotImplemented

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return -self.__radd__(other)

    def __mul__(self, other):
        match other:
            case int() | float():
                if other == 0:
                    return 0
                elif other == 1:
                    return self
                return Variable(self.name, coeff=self.coeff * other)
            case _:
                return NotImplemented

    __rmul__ = __mul__

    def __repr__(self):
        if self.coeff == 1.0:
            return self.name
        elif self.coeff == -1.0:
            return f"-{self.name}"
        else:
            return f"{self.coeff}*{self.name}"


def simplify_sum(elts):
    c = 0
    variables = {}
    for expr in elts:
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
        case Sum(elts=elts):
            return simplify_sum(elts)
        case Constraint():
            return expr.normalize()


class Sum(Expression, Comparable):
    def __init__(self, *elts):
        self.elts = elts

    def __abs__(self):
        return Abs(self)

    def __neg__(self):
        return Sum(*(-expr for expr in self.elts))

    def __add__(self, other):
        match other:
            case Sum(elts=elts):
                return Sum(*self.elts, *elts)
            case int() | float() if other == 0:
                return self
            case Variable() | Abs() | int() | float():
                return Sum(*self.elts, other)
            case _:
                return NotImplemented

    def __sub__(self, other):
        return self.__add__(-other)

    def __radd__(self, other):
        match other:
            case int() | float() if other == 0:
                if other == 0:
                    return self
                return Sum(other, *self.elts)
            case _:
                return NotImplemented

    def __rsub__(self, other):
        return -self.__radd__(other)

    def __mul__(self, other):
        match other:
            case int() | float():
                if other == 0:
                    return 0
                elif other == 1:
                    return self
                return Sum(*(other*elt for elt in self.elts))
            case _:
                return NotImplemented

    __rmul__ = __mul__

    def __repr__(self):
        # out = repr(self.elts[0])
        # for expr in self.elts[1:]:
        #     if
        elts = " + ".join(repr(x) for x in self.elts)
        return f"{elts}"


class Constraint:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def normalize(self):
        vars_left, c_left = simplify(self.left)
        vars_right, c_right = simplify(self.right)

        c = c_right - c_left
        variables = vars_left
        for name in vars_right:
            if name in variables:
                variables[name] -= vars_right[name]
            else:
                variables[name] = -vars_right[name]

        return variables, c


class Eq(Constraint):
    def __str__(self):
        return f"{self.left} = {self.right}"


class Gte(Constraint):
    def __str__(self):
        return f"{self.left} >= {self.right}"


class Lte(Constraint):
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

    def validate(self):
        ...
        #TODO: check for nesting

        # for elt in self.objective:
        #     if not isinstance(elt, Abs):
        #         continue
        #     if self.type == LinearProgram.MIN and elt.sign == -1:
        #         raise Exception("Cannot minimize a negative absolute value")
        #     if self.type == LinearProgram.MAX and elt.sign == 1:
        #         raise Exception("Cannot maximize a positive absolute value")

        # for constraint in self.constraints:
        #     for elt in self.objective:
        #         if not isinstance(elt, Abs):
        #             continue
        #         if self.type == LinearProgram.MIN and elt.sign == -1:
        #             raise Exception("Cannot minimize a negative absolute value")
        #         if self.type == LinearProgram.MAX and elt.sign == 1:
        #             raise Exception("Cannot maximize a positive absolute value")

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
        variables |= set(_vars.keys())
        for constraint in self.constraints:
            _vars, _ = simplify(constraint)
            variables |= set(_vars.keys())
        return variables

    @classmethod
    def minimize(cls, objective):
        return cls(LinearProgram.MIN, objective, [])

    @classmethod
    def maximize(cls, objective):
        return cls(LinearProgram.MAX, objective, [])

    def such_that(self, *constraints):
        self.constraints = constraints
        return self

    def optimize(self):
        all_vars = sorted(list(self._get_variables()))
        A, b, c = canonicalize(*self.as_matrix())
        value, variables = solve(LP(A, b, c, {}))
        value = value if self.type == LinearProgram.MIN else -value
        variables = variables[:len(all_vars)]
        return value, ({all_vars[i]: v for i, v in enumerate(variables)})

    def __str__(self):
        constraints = "\n".join((f"  {constraint}" for constraint in self.constraints))
        return f"{self.type} {self.objective}\ns.t.\n{constraints}"


def minimize(objective: Expression, constraints: list[Constraint]) -> tuple[float, dict[str, float]]:
    return LinearProgram.minimize(objective).such_that(*constraints).optimize()


def maximize(objective: Expression, constraints: list[Constraint]) -> tuple[float, dict[str, float]]:
    return LinearProgram.maximize(objective).such_that(*constraints).optimize()
