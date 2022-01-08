from scipy.optimize import fsolve
from functools import partial
from matplotlib import pyplot as plt
import numpy as np

class M1:
    @staticmethod
    def T(w):
        A = 0.3
        S = 1366 * w
        b = 5.67e-8
        return (S * (1 - A) / (4 * b)) ** (1 / 4)


class M2:
    @staticmethod
    def fun(p, w, a_s=0.19):
        Ta, Ts = p
        S = 1366 * w
        t_a = 0.53
        a_a = 0.3
        # a_s = 0.19
        t_ap = 0.06
        a_ap = 0.31
        b = 5.67e-8
        c = 2.7
        return ( (-t_a) * (1-a_s) * S/4 + c * (Ts-Ta) + b * Ts**4 * (1-a_ap) - b * Ta**4 , 
                 -(1-a_a-t_a+a_s*t_a) * S/4 - c * (Ts-Ta) - b * Ts**4 * (1-t_ap-a_ap) + 2 * b * Ta**4 )

    @staticmethod
    def TaTs(w):
        f = partial(M2.fun, w=w)
        return fsolve(f, (300,300))

    @staticmethod
    def TaTs_NtoG(w):
        f = partial(M2.fun, w=w)
        Ta, Ts = fsolve(f, (300,300))
        if Ts < 263:
            f = partial(M2.fun, w=w, a_s=0.62)
            return fsolve(f, (300,300))
        else:
            return Ta, Ts

    @staticmethod
    def TaTs_GtoN(w):
        f = partial(M2.fun, w=w, a_s=0.62)
        Ta, Ts = fsolve(f, (300,300))
        if Ts >= 263:
            f = partial(M2.fun, w=w)
            return fsolve(f, (300,300))
        else:
            return Ta, Ts

# 0.75

print(M1.T(1))
print()
print(M2.TaTs(0.8))
print(M2.TaTs(1))
print(M2.TaTs(1.2))

temps = [[],[]]
x = []
for w in np.arange(0.8, 1.2, 0.01):
    x.append(w)
    t = M2.TaTs(w)
    temps[0].append(t[0])
    temps[1].append(t[1])

plt.plot(x, temps[0], '.', x, temps[1], '.')
plt.title(f'Temperatury średnie względem S')
plt.xlabel('Stała słoneczna')
plt.ylabel('Średnia temperatura [k]')
plt.legend(['Temperatura atmosfery', 'Temperatura ziemi'])
plt.savefig(f"tats.png")
plt.clf()

temps = [[],[]]
x = []
for w in np.arange(0.5, 2.0, 0.01):
    x.append(w)
    temps[0].append(M1.T(w))
    temps[1].append(M2.TaTs(w)[1])

plt.plot(x, temps[0], '.', x, temps[1], '.')
plt.title(f'Średnie temperatury powierzchni względem S dla obu metod')
plt.xlabel('Stała słoneczna')
plt.ylabel('Średnia temperatura [k]')
plt.legend(['Bez uwzględnienia atmosfery', 'Z uwzględnieniem atmosfery'])
plt.savefig(f"PorMet.png")
plt.clf()

temps = [[],[]]
x = []
for w in np.arange(0.5, 2, 0.02):
    x.append(w)
    t = M2.TaTs_NtoG(w)
    temps[0].append(t[0])
    temps[1].append(t[1])

plt.plot(x, temps[0], '.', x, temps[1], '.')
plt.title(f'Temperatury średnie względem S z uwzględnieniem zmiany albedo')
plt.xlabel('Stała słoneczna')
plt.ylabel('Średnia temperatura [k]')
plt.legend(['Temperatura atmosfery', 'Temperatura ziemi'])
plt.savefig(f"NtoG.png")
plt.clf()

temps = [[],[]]
xng = []
for w in np.arange(0.5, 2, 0.02):
    xng.append(w)
    temps[0].append(M2.TaTs_NtoG(w)[1])
xgn = []
for w in np.arange(0.51, 2, 0.02):
    xgn.append(w)
    temps[1].append(M2.TaTs_GtoN(w)[1])

plt.plot(xng, temps[0], '.', xgn, temps[1], '.')
plt.title(f'Przejścia pomiędzy stanem zlodowacenia a normalnym')
plt.xlabel('Stała słoneczna')
plt.ylabel('Średnia temperatura powierzchni [k]')
plt.legend(['Normalny do Zlodowacenia', 'Zlodowacenie do Normalnego'])
plt.savefig(f"GandN.png")
plt.clf()