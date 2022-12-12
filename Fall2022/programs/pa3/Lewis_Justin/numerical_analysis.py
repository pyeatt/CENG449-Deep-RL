import numpy as np

n3_iterations = np.loadtxt("results/result_fourier_n3_iterations")
print("mean: ", np.average(n3_iterations))
print("variance: ", np.var(n3_iterations))
print("std. dev.: ", np.std(n3_iterations))
print("min", np.min(n3_iterations))

print("mean: ", np.average(n3_iterations[25:]))
print("variance: ", np.var(n3_iterations[25:]))
print("std. dev.: ", np.std(n3_iterations[25:]))
print("max: ", np.max(n3_iterations[25:]))

""" fourier n3 normal
mean:  163.3549
variance:  676.4342609900001
std. dev.:  26.008349832121223
"""

""" fourier n3 but only after learned
mean:  161.08035897435897
variance:  63.51731935831688
std. dev.:  7.9697753643573215
"""
