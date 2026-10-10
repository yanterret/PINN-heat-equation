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
# Réseaux de Neurones
#################

# création du réseaux de neurones

class MLP(nn.Module):
    '''
    entrée : 
    layer (type liste) 
    ex [2,8,8,8,1]
    '''
    def __init__(self, layers):
        super().__init__()
        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                net.append(nn.Tanh())   
        self.net = nn.Sequential(*net) # on crée le réseau couches par couches en déballant la liste

        # Initialisation avec égalisation des variance en back et foward prop 
        # pour éviter le gradient vanishing  (Xavier)

        for m in self.net:   # on prends les sorties de chaque couches    
            if isinstance(m, nn.Linear):  # ex : si tranh() = False si Linear = True (sorte de dtype) 
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
    
def model(LAYERS):
    return MLP(LAYERS).to(DEVICE)

#################
# Conditions aux limites
#################

#---------------------------------
# remarques générales:
# j'ai utilsé numpy mais c est faisable aussi avec des fonctions pytorch (à voir si il y a un gain de temps conséquend)
# on peut aussi utiliser une répartitions de points par LHS (Latin Hypercube Sampling) comme dans mon projet surrogate
# N_PDE = 5000 nombre de point dans le maillage
# N_CI  = 500  points condition initiale source
# N_CL  = 250  points par condition aux limites sur les bords
#---------------------------------

def points(seed, N_pde, N_CI, N_CL):
    '''
    entrée : 
    seed : int

    sortie :
    x_pde : (N_pde, 2) torch tensor ( valeurs en float 32 )
    x_bc : (N_CI + 2*N_CL, 2) torch tensor ( valeurs en float 32 )
    y_bc : (N_CI + 2*N_CL, 1) torch tensor ( valeurs en float 32 )
    '''
    rng = np.random.default_rng(seed)          

    # Points de collocation
    x = rng.uniform(-1.0, 1.0, size=(N_pde, 1))
    t = rng.uniform(0.0, 1.0, size=(N_pde, 1))
    x_pde = torch.tensor(np.hstack([x, t])).float()
    # points par condition aux limites sur les bords

    # conditions en t=0 
    x_ci      = np.zeros((N_CI, 2))           #shape (N_CI=500, 2)
    x_ci[:,0] = np.linspace(-1, 1, N_CI)   
    x_ci[:,1] = 0.0                        # t=0
    y_ci      = np.sin(np.pi * x_ci[:,0:1])   #shape (500, 1)

    # conditions en x=-1
    x_cl_l      = np.zeros((N_CL, 2))       #shape (N_CL = 250 , 2)
    x_cl_l[:,0] = -1.0
    x_cl_l[:,1] = np.linspace(0, 1, N_CL)
    y_cl_l      = np.zeros((N_CL, 1))       #shape (250 , 1) On mets une température de 0

    # --- CL droite : x=+1, T=0 ---
    x_cl_r      = np.zeros((N_CL, 2))     #shape (250, 2)
    x_cl_r[:,0] = 1.0
    x_cl_r[:,1] = np.linspace(0, 1, N_CL)
    y_cl_r      = np.zeros((N_CL, 1))     #shape (250, 1) de meme


    x_bc_np = np.vstack([x_ci, x_cl_l, x_cl_r])     #shape (1000, 2) 
    y_bc_np = np.vstack([y_ci, y_cl_l, y_cl_r])     #shape (1000, 1)

    x_bc = torch.tensor(x_bc_np).float()        #transforme le float 64 en 32
    y_bc = torch.tensor(y_bc_np).float()

    return x_pde.to(DEVICE), x_bc.to(DEVICE), y_bc.to(DEVICE)


def source_fn(x_torch, t_torch):
    return (np.pi**2 - 1) * torch.exp(-t_torch) * torch.sin(np.pi * x_torch)


#################
# Fonction Cout 
#################


def loss_bc(model, x_bc, y_bc):
    """
    entrée :
    x_bc : (N_CI + 2*N_CL, 2) torch tensor ( valeurs en float 32 ) : points sur les bords
    y_bc : (N_CI + 2*N_CL, 1) torch tensor ( valeurs en float 32 ) : CI ex 0 aux bords et sin(pi*x) sur un coté 

    sortie :
    loss_bc : int 
    """
    pred = model(x_bc)
    return torch.mean( (pred - y_bc)**2) # torch.float32 torch.Size([]) 


def loss_pde(model, x_pde, alpha=1.0, source_fn=None):
    """
    entrée :
     x_pde : (N_pde, 2) torch tensor ( valeurs en float 32 )

    sortie :
    loss_pde : int 
    """
    x_torch = x_pde[:, 0:1].float().requires_grad_(True)    #torch.Size([5000, 1]) torch.float32
    t_torch = x_pde[:, 1:2].float().requires_grad_(True)    #torch.Size([5000, 1])

    # on obtient y(x,t)
    T_torch = torch.cat([x_torch, t_torch], dim=1)    #torch.Size([5000, 2]) comme concatenate sur numpy
    T = model(T_torch)  #torch.Size([5000, 1])

    # grad_outputs=torch.ones_like(T) permet d'avoir en sortie un tenseur de torch.Size([5000, 1]) sinon celle d'apres ne marche pas
    dT_dt = torch.autograd.grad(T, t_torch,grad_outputs=torch.ones_like(T),create_graph=True)[0]   #torch.Size([5000, 1])
    dT_dx = torch.autograd.grad(T, x_torch,grad_outputs=torch.ones_like(T),create_graph=True)[0] #torch.Size([5000, 1])
    dT_dx2 = torch.autograd.grad(dT_dx, x_torch,grad_outputs=torch.ones_like(T),create_graph=True)[0]  #torch.Size([5000, 1])

    # définition du terme source

    if source_fn == None:
        f=0.0
    else:
        f = source_fn(x_torch, t_torch)  #torch.Size([5000, 1])

    # résidu ( équation de la chaleur)
    
    residu = dT_dt - alpha * dT_dx2 - f   #torch.Size([5000, 1])
    return torch.mean(residu ** 2)   # torch.float32 torch.Size([]) 


def total_loss(model, x_bc, y_bc, x_pde,
               w_bc, w_pde,
               alpha=1.0, source_fn=None):
    """
    entrée :
    ...
    w_bc w_pde, les poids d'importances des loss à prendre en compte

    sortie :
    total_loss : int 
    """
    l_bc  = loss_bc(model, x_bc, y_bc)
    l_pde = loss_pde(model, x_pde, alpha=alpha, source_fn=source_fn)
    loss  = w_bc * l_bc + w_pde * l_pde  # torch.float32 torch.Size([]) 
    return loss, l_bc, l_pde

#################
# fonction de test
#################
N_test = 200

def u_exact(x, t):
    """
    entrée :
    x : array ou int 
    t : array ou int 

    sortie :
    array ou int
    """
    return np.exp(-t) * np.sin(np.pi * x)

def test_mesh(N_test,u_exact):
    """
    entrée :
    N_test : int
    u_exact : fonction solution analytique de notre modéle

    sortie :
    err_l2 : int 
    """
    x_test = np.linspace(-1, 1, N_test)
    t_test = np.linspace(0,  1, N_test)

    X, T_grid = np.meshgrid(x_test, t_test) # création de 2 matrices N_test*N_test (la grille)

    XT = torch.FloatTensor(np.stack([X.flatten(), T_grid.flatten()], axis=1)).to(DEVICE) # (N_test**2, 2)
    # résultat
    with torch.no_grad():
        T_pred = model(XT).cpu().numpy().reshape(N_test, N_test) 

    T_exact = u_exact(X, T_grid)

    err_l2 = (np.sqrt(np.mean((T_pred - T_exact)**2))/ np.sqrt(np.mean(T_exact**2)))

    return err_l2

