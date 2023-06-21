

class Literal:
    pass


class Variable(Literal):
    def __init__(self, name, coeff=1, lower_bound=0):
        self.name = name
        self.coeff = coeff

    def __neg__(self):
        return Variable(name=self.name, coeff=-self.coeff)
    
    def __add__(self, other):
        match other:
            case Variable(name=name, coeff=coeff) if name == self.name:
                return Variable(name=name, coeff=self.coeff+coeff)
            case Variable():
                return Sum(self, other)
            case int() | float():
                return Sum(self, other)
            case _:
                return NotImplemented
                # raise TypeError("Can only add numbers or variables")
    
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
        else:
            return f"{self.coeff}*{self.name}"



def flatten(expr):
    match expr:
        case Sum(exprs=exprs):
            flat = []
            for expr in exprs:
                match expr:
                    case list():
                        flat += expr
                    case _:
                        flat.append(expr)
            return flat
        case _:
            return expr


class Expression:
    pass


class Sum(Expression):
    def __init__(self, *exprs):
        self.exprs = exprs

    def __eq__(self, other):
        return Eq(self, other)
    
    def __add__(self, other):
        match other:
            case Sum(exprs=exprs):
                return Sum(*self.exprs, *exprs)
            case Variable() | int() | float():
                return Sum(*self.exprs, other)
            case _:
                return NotImplemented

    def __radd__(self, other):
        match other:
            case Sum(exprs=exprs):
                return Sum(*exprs, *self.exprs)
            case Variable() | int() | float():
                return Sum(other, *self.exprs)
            case _:
                return NotImplemented
    
    def __repr__(self):
        exprs = ", ".join(repr(x) for x in self.exprs)
        return f"({exprs})"


class Comparison(Expression):
    pass


class Eq(Comparison):
    def __init__(self, left, right):
        self.left = left
        self.right = right


    def normalize(self):
        left = flatten(self.left)
        right = flatten(self.right)

        # print("L", left)
        # print("R", right)

        variables = {}
        c = 0

        if not isinstance(left, list):
            left = [left]

        for expr in left:
            match expr:
                case Variable():
                    if expr.name not in variables:
                        variables[expr.name] = expr
                    else:
                        variables[expr.name] += expr
                case int() | float():
                    c -= expr

        if not isinstance(right, list):
            right = [right]

        for expr in right:
            match expr:
                case Variable():
                    if expr.name not in variables:
                        variables[expr.name] = expr
                    else:
                        variables[expr.name] -= expr
                case int() | float():
                    c += expr

        keys = sorted(list(variables.keys()))
        return [variables[k] for k in keys], c


class LinearProgram:
    MIN = 'min'
    MAX = 'max'

    def __init__(self, type, objective, constraints):
        self.type = type
        self.objective = objective
        self.constraints = constraints

    def compile(self):
        ...

    def _get_variables(self):
        variables = set()


# X + (Y*2 - 4) + X = 23

X = Variable("X")
Y = Variable("Y")

print(X)
print(Y*2 - 4)
print(X + (Y*2 - 4) + X)

eq = ((X + (Y*2 - 4) + X) == 23)

print(eq.left)
print(eq.right)

print("=====")
print(eq.normalize())


