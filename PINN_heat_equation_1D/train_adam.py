#################
# Libraries
#################
import torch
import torch.nn as nn
import numpy as np
import os

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
            print(f'Epoch {epoch:5d} | Loss: {loss.item():.4e} | BC: {l_bc.item():.4e} | PDE: {l_pde.item():.4e}| Temps_tot: {time.perf_counter() - t_tot :.4e} ')

