<p align="center">
    <img src="https://raw.githubusercontent.com/tomasr8/pivotal/master/logo.svg">
</p>

# No fuss Linear Programming solver

```python
from pivotal import minimize, maximize, Variable

x = Variable("x")
y = Variable("y")
z = Variable("z")

objective = 2*x + y + 3*z
constraints = (
    x - y == 4,
    y + 2*z == 2
)

minimize(objective, constraints)
# -> value: 11.0
# -> variables: {'x': 4.0, 'y': 0.0, 'z': 1.0}

maximize(objective, constraints)
# -> value: 14.0
# -> variables: {'x': 6.0, 'y': 2.0, 'z': 0.0}
```

## About

`Pivotal` is not aiming to compete with commerical solvers like Gurobi. Rather, it is aiming to simplify the process of creating and solving linear programs thanks to its very simple and intuitive API. The solver itself uses a 2-phase Simplex algorithm.

## Installation

Python >=3.10 is required.

Install via pip:

```bash
pip install pivotal-solver
```

## API

### Variables

`Variable` instances implement `__add__`, `__sub__` and other magic methods, so you can use them directly in expressions such as `2*x + 10 - y`.

Here are some examples of what you can do with them:

```python
x = Variable("x")
y = Variable("y")
z = Variable("z")

2*x + 10 - y
x + (y - z)*10
-x
-(x + y)
sum([x, y, z])

X = [Variable(f"x{i}") for i in range(5)]
sum(X)
```

Note that variables are considered equal if they have the same name, so
for examples this expression:

```python
Variable("x") + 2 + Variable("x")
```

will be treated as simply `2*x+2`.

The first argument to `minimize` and `maximize` is the objective function which must be either a single variable or a linear combination as in the example above.

### Constraints

There are three supported constraints: `==` (equality), `>=` (greater than or equal) and `<=` (less than or equal). You create a constraint simply by using these comparisons in expressions involving `Variable` instances. For example:

```python
x = Variable("x")
y = Variable("y")
z = Variable("z")

x == 4
2*x - y == z + 7
y >= -x + 3*z
x <= 0
```

There is no need to convert your constraints to the canonical form which uses only equality constraints. This is done automatically by the solver.

`minimize` and `maximize` expect a list of constraints as the second argument.

### Output

The return value of `minimize` and `maximize` is a 2-tuple containing the value of the objective function and a dictionary of variables and their values.

The functions may raise `pivotal.Infeasible` if the program is over-constrained (no solution exists) or `pivotal.Unbounded` if the program is under-constrained (the objective can be made arbitrarily small).

```python
from pivotal import minimize, maximize, Variable, Infeasible

x = Variable("x")
y = Variable("y")

objective = 2*x + y
constraints = (
    x + 2*y == 4,
    x + y == 10
)

try:
    minimize(objective, constraints)
except Infeasible:
    print("No solution")
```

## Limitations

- Currently, all variables are assumed to be positive

## TODO (Contributions welcome)

- (WIP) Arbitrary variable bounds, e.g. `a <= x <= b`
- (WIP) Support for absolute values
- Setting tolerance & max number of iterations
- MILP solver with branch & bound


## Development

### Setting up

```python
git clone https://github.com/tomasr8/pivotal.git
cd pivotal
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Running tests

```python
pytest
```