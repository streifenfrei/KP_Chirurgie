import time
import numpy as np


def gaussian_function_slow(x, sigma):
    a = 1 / (sigma * np.sqrt(2 * np.pi))
    gaussian = np.exp(-(x / (2 * (sigma ** 2))))
    return a * gaussian


def gaussian_function_fast(x):
    gaussian = np.exp(-(x / (2 * (sigma ** 2))))
    return a * gaussian


class GaussianFunctionClass():

    def __init__(self, sigma):
        self.sigma = sigma
        self.a = 1 / (sigma * np.sqrt(2 * np.pi))

    def __call__(self, x):
        gaussian = np.exp(-(x / (2 * (self.sigma ** 2))))
        return self.a * gaussian


if __name__ == '__main__':
    sigma = 5
    a = 1 / (sigma * np.sqrt(2 * np.pi))

    t = 1000000

    # slow
    start = time.time()
    for i in range(t):
        gaussian_function_slow(i, sigma)
    duration = (time.time() - start)
    print("slow gaussian took: {0} seconds".format(duration))

    # fast
    start = time.time()
    for i in range(t):
        gaussian_function_fast(i)
    duration = (time.time() - start)
    print("fast gaussian took: {0} seconds".format(duration))

    # class
    gaussian_function_class = GaussianFunctionClass(sigma)
    start = time.time()
    for i in range(t):
        gaussian_function_class(i)
    duration = (time.time() - start)
    print("class gaussian took: {0} seconds".format(duration))
