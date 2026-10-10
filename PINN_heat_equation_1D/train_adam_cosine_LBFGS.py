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
w_bc=100.0
w_pde=1.0

epochs_tot=3000
N_test = 200

SEEDS = [x for x in range(1, 11)]
architecture  = [2, 32, 32, 32, 1]

# paramétres lbfg
epochs_lbfg = 100
eta_min = 1e-5
T_max = epochs_tot
lr_lbfg=0.1
max_iter=20
line_search_fn='strong_wolfe'
RESAMPLE_FREQ_lbfg = 20

# paramétres adam
lr=1e-3
RESAMPLE_FREQ = 500
epochs_adam = epochs_tot - epochs_lbfg



#################
# Fonction inhérente à l'utilisation de lbfgs
#################
# recalcul la backprop pour que lbfgs choississe la bonne valeur

def closure():
    optimizer_lbfgs.zero_grad()
    loss, _, _ = total_loss(model, x_bc_d, y_bc_d, x_pde_d, source_fn=source_fn)
    loss.backward()
    return loss

#################
# entrainement à seed fixe
#################

history = {'loss': [], 'l_bc': [], 'l_pde': []}

for seed in SEEDS:

    torch.manual_seed(seed)
    np.random.seed(seed)
    x_pde_d, x_bc_d, y_bc_d = points(seed, N_pde, N_CI, N_CL)

    #################
    # définition du modéle et des optimizers
    #################

    model= MLP(architecture).to(DEVICE)
    optimizer= torch.optim.Adam(model.parameters(), lr)

    # scheduler = technique cosine
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max,eta_min)
    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(),lr_lbfg,max_iter, line_search_fn )# max iter nombre d essais par step(), line_search_fn test pour avoir le bon lr 

    ################## 
    # boucle d'entrainement adam 
    ##################
    
    for epoch in range(epochs_adam):
        optimizer.zero_grad()
        loss, l_bc, l_pde = total_loss(model, x_bc_d, y_bc_d, x_pde_d,w_bc, w_pde, source_fn=source_fn)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        history['loss'].append(loss.item())
        history['l_bc'].append(l_bc.item())
        history['l_pde'].append(l_pde.item())
        if epoch % RESAMPLE_FREQ == 0:
            print(f'Epoch {epoch:5d} | Loss: {loss.item():.4e} | BC: {l_bc.item():.4e} | PDE: {l_pde.item():.4e}')


    ################## 
    # boucle d'entrainement lbfgs 
    ##################

    for epoch in range(epochs_lbfg):

        optimizer_lbfgs.step(closure)
        loss, l_bc, l_pde = total_loss(model, x_bc_d, y_bc_d, x_pde_d, w_bc, w_pde,source_fn=source_fn)
        
        history['loss'].append(loss.item()) #.item() pour donner un int 
        history['l_bc'].append(l_bc.item())
        history['l_pde'].append(l_pde.item())
        if epoch % RESAMPLE_FREQ_lbfg == 0:
            print(f'Epoch {epoch:5d} | Loss: {loss.item():.4e} | BC: {l_bc.item():.4e} | PDE: {l_pde.item():.4e}')
