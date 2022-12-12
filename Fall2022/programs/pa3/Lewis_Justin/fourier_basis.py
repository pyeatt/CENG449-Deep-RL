from basis import Basis
import numpy as np
from itertools import product

# phi_i (x) = cos(i pi x) for i = 0, ..., n
class FourierBasis(Basis):
    def __init__(self, d, n):
        super().__init__(d, n)
        self.c = np.pi * np.array(list(product(range(n+1), repeat=d)))

    def apply(self, x):
        toApply = [ (x[0] + 1.2 )/1.7,
                    (x[1] + 0.07 )/0.14,
                    (x[2] + 1)/2
                ]
        return np.cos(np.dot(self.c, toApply))

    def getC(self):
        return np.array(list(product(range(self.n+1), repeat=self.d)))


if __name__ == "__main__":
    dim = 2
    n = 4
    b = FourierBasis(dim, n)
    
    print(np.shape(b.c)) # len(9) = (n)^d
