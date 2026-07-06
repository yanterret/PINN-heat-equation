import torch
import numpy as np
import matplotlib.pyplot as plt

def u_exact(x, t):
    return np.exp(-t) * np.sin(np.pi * x)

# grille de test 200×200
x_test = np.linspace(-1, 1, 200)
t_test = np.linspace(0,  1, 200)

X, T_grid = np.meshgrid(x_test, t_test) # création de 2 matrices 200*200 (la grille)

XT = torch.FloatTensor(np.stack([X.flatten(), T_grid.flatten()], axis=1)).to(DEVICE) # (40000, 2)
# résultat
with torch.no_grad():
    T_pred = model(XT).cpu().numpy().reshape(200, 200) 

T_exact = u_exact(X, T_grid)

err_l2 = (np.sqrt(np.mean((T_pred - T_exact)**2))/ np.sqrt(np.mean(T_exact**2)))
