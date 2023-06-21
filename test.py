import numpy as np


c = np.block([
    [np.eye(1), np.atleast_2d(1)],
    [np.eye(1), np.eye(1)]
])

print(c)
