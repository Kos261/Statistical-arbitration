import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import cumsum


def generate_poisson_1(lambda_1, time_step):
    n = int(1. / time_step)  # liczba kroków
    x = np.zeros(n)
    x[np.random.rand(n) <= lambda_1 * time_step] = 1
    return x

if __name__ == '__main__':
    poiss = generate_poisson_1(20, 0.001)
    cum = np.cumsum(poiss)

    plt.subplot(3, 1, 1)
    plt.plot(cum)

    plt.subplot(3, 1, 2)
    plt.stem(poiss)
    plt.title("Lambda = 20")

    n=10000
    histogram = np.zeros(n)
    for i in range(n):
        poiss = generate_poisson_1(3, 0.001)
        histogram[i] = np.sum(poiss)

    plt.subplot(3, 1, 3)
    plt.hist(histogram)
    plt.title("Lambda = 3")

    plt.show()