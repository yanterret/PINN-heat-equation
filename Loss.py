import torch
import numpy as np
import matplotlib.pyplot as plt

from CI import x_bc
from CI import y_bc
from CI import x_pde
from PINN import MLP

#Loss MSE sur les CI.

def loss_bc(model, x_bc, y_bc):
    """
    entrée :
    x_bc : (N, 2) type torch tensor : points sur les bords
    y_bc : (N, 1) type torch tensor : CI ex 0 aux bords et sin(pix) sur un coté 

    sortie :
    loss_bc : int 

    """
    pred = model(x_bc)
    return torch.mean( (pred - y_bc)**2) # torch.float32 torch.Size([]) 

def loss_pde(model, x_pde, alpha=1.0, source_fn=None):
    """
    entrée :
    x_pde : (N, 2) type torch tensor : points de collocation intérieurs

    sortie :
    loss_pde : int 

    """
    x_torch = x_pde[:, 0:1].float().requires_grad_(True)    #torch.Size([5000, 1]) torch.float32
    t_torch = x_pde[:, 1:2].float().requires_grad_(True)    #torch.Size([5000, 1])

    # on obtient y(x,t)
    T_torch = torch.cat([x_torch, t_torch], dim=1)    #torch.Size([5000, 2]) comme concatenate sur numpy
    T = model(T_torch)  #torch.Size([5000, 1])

    # dérivée premiere en fonction du temps
    # grad_outputs=torch.ones_like(T) permet d'avoir en sortie un tenseur de torch.Size([5000, 1]) sinon celle d'apres ne marche pas
    dT_dt = torch.autograd.grad(T, t_torch,grad_outputs=torch.ones_like(T),create_graph=True)[0]   #torch.Size([5000, 1])

    # dérivée premiere en fonction de l'espace
    dT_dx = torch.autograd.grad(T, x_torch,grad_outputs=torch.ones_like(T),create_graph=True)[0] #torch.Size([5000, 1])

    # dérivée seconde en fonction de l'espace
    dT_dx2 = torch.autograd.grad(dT_dx, x_torch,grad_outputs=torch.ones_like(T),create_graph=True)[0]  #torch.Size([5000, 1])

    # Étape 6 : terme source
    if source_fn == None:
        f=0.0
    else:
        f = source_fn(x_torch, t_torch)  #torch.Size([5000, 1])

    # résidu ( équation de la chaleur)
    
    residu = dT_dt - alpha * dT_dx2 - f   #torch.Size([5000, 1])
    return torch.mean(residu ** 2)   # torch.float32 torch.Size([]) 

def total_loss(model, x_bc, y_bc, x_pde,
               w_bc=100.0, w_pde=1.0,
               alpha=1.0, source_fn=None):
    """
    entrée :
    explicité avant
    w_bc w_pde, les poids d'importances des loss à prendre en compte

    sortie :
    total_loss : int 
 
    """
    l_bc  = loss_bc(model, x_bc, y_bc)
    l_pde = loss_pde(model, x_pde, alpha=alpha, source_fn=source_fn)
    loss  = w_bc * l_bc + w_pde * l_pde  # torch.float32 torch.Size([]) 
    return loss, l_bc, l_pde