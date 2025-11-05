import numpy as np
import matplotlib.pyplot as plt
from numpy import random

def main():
    lbd = 20.0
    n = 100
    poiss = random.exponential(scale=1/lbd, size=n)
    event_times = np.cumsum(poiss)

    t = np.linspace(0, event_times[-1], 500)
    Nt = [np.sum(event_times <= ti) for ti in t] #How many less than ti

    plt.figure(figsize=(8, 6))

    # plt.subplot(2, 1, 1)
    # plt.plot(event_times, np.arange(1, n + 1), drawstyle='steps-post')
    # plt.title("Proces zliczający N(t) dla $\lambda$ = 20")
    # plt.xlabel("Czas")
    # plt.ylabel("Liczba zdarzeń")

    plt.subplot(2, 1, 1)
    plt.plot(Nt, t, drawstyle='steps-post')
    plt.title("Proces zliczający N(t) dla $\lambda$ = 20")
    plt.xlabel("Czas")
    plt.ylabel("Liczba zdarzeń")

    plt.subplot(2, 1, 2)
    plt.vlines(event_times[:100], ymin=0, ymax=1)
    plt.title("Pierwsze 50 momentów wystąpienia zdarzeń")
    plt.yticks([])
    plt.xlabel("Czas")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()