
#---------------------------------
# remarques générales:
# ce code a servi et à été conçu pour comparer plusieurs architectures de réseaux de neurones
# il ne fait pas donc pas de bilan partiels tout les n epochs car servait à comparer uniquement les résulats post entrainement 
#---------------------------------

#################
# Libraries
#################

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#################
# Configuration de l'environnement de calcul 
#################

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#################
# importation des fonctions
#################

from PINN_heat_equation import MLP,total_loss,source_fn,u_exact,points,test_mesh

#################
# paramétres de la simulation
#################

N_pde=5000
N_CI=500
N_CL=250
epochs=3000
N_test = 200
lr=1e-3
w_bc=100.0
w_pde=1.0
SEEDS = [x for x in range(1, 11)]
architectures = {
    "4c x 8" : [2,8,8,8,8,1],
    "5c x 8" : [2,8,8,8,8,8,1],
}

#################
# fonction entrainement
#################

def train(layers, seed, epochs):
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_pde_d, x_bc_d, y_bc_d = points(seed, N_pde, N_CI, N_CL)      
    model = MLP(layers).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss, _, _ = total_loss(model, x_bc_d, y_bc_d, x_pde_d,
                                w_bc, w_pde, source_fn=source_fn)
        loss.backward()
        optimizer.step()

    err_model = test_mesh(N_test,u_exact)

    return err_model

#################
# Visualisation des résultats
#################

resultats = {}

for nom, layers in architectures.items():
    erreurs = np.array([train(layers, s) for s in SEEDS])
    resultats[nom] = erreurs
    n_rates = (erreurs > 0.02).sum()

    print(f"{nom}")
    print(f"  moyenne : {erreurs.mean()}   médiane : {np.median(erreurs)}")
    print(f"  std     : {erreurs.std()}   min/max : {erreurs.min()} / {erreurs.max()}")
    print(f"  runs ratés (>2%) : {n_rates} / {len(SEEDS)}")