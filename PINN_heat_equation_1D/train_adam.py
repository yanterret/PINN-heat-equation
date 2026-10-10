#################
# Libraries
#################
import torch
import torch.nn as nn
import numpy as np
import time
import os

#################
# Configuration de l'environnement de calcul 
#################
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#################
# importation des fonctions
#################
from PINN_heat_equation import MLP,total_loss,source_fn,u_exact,points,test_mesh,train

#################
# paramétres de la simulation
#################
N_pde=5000
N_CI=500
N_CL=250
w_bc=100.0
w_pde=1.0
epochs=3000
N_test = 200
SEEDS = [x for x in range(1, 11)]
architecture  = [2, 32, 32, 32, 1]

# paramétres adam
lr=1e-3
RESAMPLE_FREQ = 500

#################
# entrainement à seed fixe
#################

history = {'loss': [], 'l_bc': [], 'l_pde': []}
resultats = {}
for seed in SEEDS:

    torch.manual_seed(seed)
    np.random.seed(seed)
    x_pde_d, x_bc_d, y_bc_d = points(seed, N_pde, N_CI, N_CL)   

    #################
    # définition du modéle et de l'optimizer
    ##################    
    model = MLP(architecture).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    ################## 
    # boucle d'entrainement 
    ##################
    t_tot = time.perf_counter()    
    for epoch in range(epochs):

        optimizer.zero_grad()
        loss, l_bc, l_pde = total_loss(model, x_bc_d, y_bc_d, x_pde_d,w_bc, w_pde, source_fn=source_fn)
        loss.backward()
        optimizer.step()

        history['loss'].append(loss.item())
        history['l_bc'].append(l_bc.item())
        history['l_pde'].append(l_pde.item())

        if epoch % RESAMPLE_FREQ == 0:
            print(f'Epoch {epoch:5d} | Loss: {loss.item():.4e} | BC: {l_bc.item():.4e} | PDE: {l_pde.item():.4e}')

    #################
    # Visualisation des résultats
    #################
    T_final = time.perf_counter() - t_tot
    err_model = test_mesh(N_test,u_exact,model)
    resultats[seed] = err_model
    n_rates = (err_model > 0.02).sum()
    print(f'Temps_tot: {T_final :.4e}')
    print(f"  moyenne : {err_model.mean()}   médiane : {np.median(err_model)}")
    print(f"  std     : {err_model.std()}   min/max : {err_model.min()} / {err_model.max()}")
    print(f"  runs ratés (>2%) : {n_rates} / {len(SEEDS)}")

#################
# Sauvegarde des résultats
#################
os.makedirs("resultats", exist_ok=True)

with open("resultats/train_adam.csv", "w") as f:
    f.write("moyenne,mediane,std,min,max,runs_rates")
    f.write("\n")
    for erreurs in resultats.items():
        f.write(f",{erreurs.mean():.6f}")
        f.write(f",{np.median(erreurs):.6f}")
        f.write(f",{erreurs.std():.6f}")
        f.write(f",{erreurs.min():.6f}")
        f.write(f",{erreurs.max():.6f}")
        f.write(f",{(erreurs > 0.02).sum()}")
        f.write("\n")
