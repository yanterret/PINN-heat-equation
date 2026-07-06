import numpy as np
from scipy.integrate import odeint

nx = 200                          # nombre de points en x 
Vx = np.linspace(-1, 1, nx + 1)   # vecteur d espace
Vt = np.linspace( 0, 1, nx + 1)   # vecteur de temps
h  = Vx[1] - Vx[0]                 # pas d'espace

def source(x, t):
    return (np.pi**2 - 1) * np.exp(-t) * np.sin(np.pi * x)

def exact(x, t):
    return np.exp(-t) * np.sin(np.pi * x)

# discrétisation
def discretisation(T, t):
    dT_dx2 = np.zeros_like(T)
    # On applique l'équation seulement aux points de collocations
    # T[0] et T[-1] restent à 0 (CI).
    dT_dx2[1:-1] = (T[2:] - 2*T[1:-1] + T[:-2]) / h**2
    return dT_dx2 + source(Vx, t)

# CI
T0 = np.sin(np.pi * Vx)

# méthode odeint
S = odeint(discretisation, T0, Vt) # shape (nt, nx+1)

X, T = np.meshgrid(Vx, Vt, indexing='xy')

erreur = np.sqrt(np.sum((S - exact(X, T))**2)) / np.sqrt(np.sum(exact(X, T)**2))