import matplotlib.pyplot as plt
import numpy as np

def v(t,m,r,P):
    return P * ((1 + r/m)) ** (m * t)

t = np.linspace(0,100,100)
r = 0.02
P = 1000
m_list = [1,10,50,365]

plt.figure(figsize=(10, 6))
for m in m_list:
    yi = v(t, m, r, P)
    plt.plot(t, yi, label=f"m = {m} (co {365//m if m<365 else 1} dni)", alpha=0.7)

plt.plot(t, P * np.exp(t * r),
         color="red",
         linestyle="--",
         label="Granica (Kapitalizacja ciągła $e^{rt}$)")

plt.title(f"Zbieżność procentu składanego do $e$ (r={r*100}%)")
plt.xlabel("Czas (lata)")
plt.ylabel("Wartość inwestycji")
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.show()

