from basis import Basis
import numpy as np
from itertools import product

class PolynomialBasis(Basis):
    def __init__(self, d, n):
        super().__init__(d, n)
        self.c = np.array(list(product(range(n+1), repeat=d)))

    def getC(self):
        return self.c

    def apply(self, x):
#         x = [ (x[0] + 1.2 )/1.7,
#                     (x[1] + 0.07 )/0.14,
#                     (x[2] + 1)/2
#                 ]
        phi = [np.prod([x[_d]**self.c[_n][_d] for _d in range(self.d)]) for _n in
                range(len(self.c))]
        return phi

if __name__ == "__main__":
    n = 3
    d = 2
    b = PolynomialBasis(d, n)
    coefficients = b.c
    x = np.array([2, 3])
    for c in coefficients:
        print(f"x^{c[0]}y^{c[1]}", end=" | ")
        print(f"{x[0]**c[0] * x[1]**c[1]}")
    print(b.apply(x))
