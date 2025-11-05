import numpy as np
from matplotlib import pyplot as plt

coefs = np.array([- 1/3, 2, 0, 0])

X = np.linspace(-2, 4, 100)
polynomial = np.polyval(coefs, X)

result = np.polyfit(X, polynomial, 3)
plt.plot(X, result[3] + X**1 * result[2] + X**2 * result[1] +  X**3 * result[0])

result = np.polyfit(X, polynomial, 2)
plt.plot(X, X**0 * result[2] + X**1 * result[1] +  X**2 * result[0])

result = np.polyfit(X, polynomial, 1)
plt.plot(X, X**0 * result[1] +  X**1 * result[0])

plt.plot(X, polynomial, linestyle='--')
plt.show()
