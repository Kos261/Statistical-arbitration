# Lab I.80 Punktowa estymacja gęstości rozkładu (normalnego)
# Urządzenie parametryzowane wielkością $\mu$ produkuje części,
# których średnica ma rozkład normalny o parametrach $N(\mu, 0.05)$.
# Część uważa się za dobrą, jeśli jej średnica mieści się w przedziale $(20.15, 20.25)$.
# W jaki sposób dobrać parametr $\mu$, aby prawdopodobieństwo wykonania części
# niespełniających powyższego kryterium jakości było najmniejsze?
# Napisz program znajdujący rozwiązanie.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm #cumulative distribution function

def prob_good(mu, sigma=0.05, a=20.15, b=20.25):
    result = norm.cdf((b-mu)/sigma) - norm.cdf((a-mu)/sigma)
    return result

if __name__ == "__main__":
    # mu_star = 20.20
    sigma = 0.05
    # N = np.random.normal(mu_star, 0.05,size=100)
    # x = np.linspace(mu_star - 3 * sigma, mu_star + 3 * sigma, 100)
    # plt.bar(x,N)
    # plt.show()


    params = np.linspace(20.0,20.4,1000)
    probabilities = [prob_good(mu, sigma=sigma) for mu in params]
    mu_star = params[np.argmax(probabilities)]
    max_prob = max(probabilities)

    print("Optimal mu: " , mu_star)
    print("Best probability: " , str(max_prob))

