import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    """
    MLP pour PINN.
    layers : ex [2, 32, 32, 32, 1]
             entrée 2D → 3 couches cachées → sortie scalaire
    """
    def __init__(self, layers):
        super().__init__()
        
        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                net.append(nn.Tanh())    #tanh en fonction d activation partout sauf à la fin 
                
        self.net = nn.Sequential(*net)

        for m in self.net:         # on égalise les variance en back et foward prop pour éviter le gradient vanishing avec les données de départ
            if isinstance(m, nn.Linear):  # on prends juste les sorties de chaque couches
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        return self.net(x)

N_PDE = 5000

x = np.random.uniform(-1.0, 1.0, size=(N_PDE, 1)) #shape (N_PDE ,1)
t = np.random.uniform(0, 1.0, size=(N_PDE, 1))    #shape (N_PDE ,1)
x_pde_np = np.hstack([x,t])      #shape (N_PDE ,2) float64
x_pde = torch.tensor(x_pde_np).float()    #torch.Size([5000, 2]) torch.float32

N_CI  = 500   # points condition initiale
N_CL  = 250   # points par condition aux limites

# --- Condition initiale : t=0, x ∈ [-1,1] ---
x_ci      = np.zeros((N_CI, 2))               #shape (N_CI=500, 2)
x_ci[:,0] = np.linspace(-1, 1, N_CI)   # x
x_ci[:,1] = 0.0                        # t=0
y_ci      = np.sin(np.pi * x_ci[:,0:1])   #shape (500, 1)

# --- CL gauche : x=-1, T=0 ---
x_cl_l      = np.zeros((N_CL, 2))       #shape (N_CL = 250 , 2)
x_cl_l[:,0] = -1.0
x_cl_l[:,1] = np.linspace(0, 1, N_CL)
y_cl_l      = np.zeros((N_CL, 1))       #shape (250 , 1)

# --- CL droite : x=+1, T=0 ---
x_cl_r      = np.zeros((N_CL, 2))     #shape (250, 2)
x_cl_r[:,0] = 1.0
x_cl_r[:,1] = np.linspace(0, 1, N_CL)
y_cl_r      = np.zeros((N_CL, 1))     #shape (250, 1)

# --- Concaténation ---
x_bc_np = np.vstack([x_ci, x_cl_l, x_cl_r])     #shape (1000, 2) 
y_bc_np = np.vstack([y_ci, y_cl_l, y_cl_r])     #shape (1000, 1)

x_bc = torch.tensor(x_bc_np).float()        #transforme le float 64 en 32 ( format torch)
y_bc = torch.tensor(y_bc_np).float()

def loss_bc(model, x_bc, y_bc):
    """
    Loss MSE sur conditions initiales et aux limites.
    x_bc : (N, 2) — points sur les bords
    y_bc : (N, 1) — valeurs connues (0 sur CL, sin(πx) sur CI)
    """
    pred = model(x_bc)
    return torch.mean( (pred - y_bc)**2)

def loss_pde(model, x_pde, alpha=1.0, source_fn=None):
    """
    Loss résidu EDP : ∂T/∂t - alpha*∂²T/∂x² - f(x,t) = 0
    x_pde : (N, 2) — points de collocation intérieurs
    """

    x_torch = x_pde[:, 0:1].float().requires_grad_(True)    #torch.Size([5000, 1]) torch.float32
    t_torch = x_pde[:, 1:2].float().requires_grad_(True)    #torch.Size([5000, 1])

    # Étape 2 : forward pass
    T_torch = torch.cat([x_torch, t_torch], dim=1)    #torch.Size([5000, 2])
    T = model(T_torch)  #torch.Size([5000, 1])

    # Étape 3 : ∂T/∂t
    dT_dt = torch.autograd.grad(T, t_torch,grad_outputs=torch.ones_like(T),create_graph=True)[0]   #torch.Size([5000, 1])

    # Étape 4 : ∂T/∂x
    dT_dx = torch.autograd.grad(T, x_torch,grad_outputs=torch.ones_like(T),create_graph=True)[0] #torch.Size([5000, 1])

    # Étape 5 : ∂²T/∂x²
    d2T_dx2 = torch.autograd.grad(dT_dx, x_torch,grad_outputs=torch.ones_like(T),create_graph=True)[0]  #torch.Size([5000, 1])

    # Étape 6 : terme source
    f = source_fn(x_torch, t_torch) if source_fn else 0.0  #torch.Size([5000, 1])

    # Étape 7 : résidu et loss
    residu = dT_dt - alpha * d2T_dx2 - f   #torch.Size([5000, 1])
    return torch.mean(residu ** 2)   # torch.float32 torch.Size([]) (scalaire)

# --- Terme source Example 1 ---
def source_fn(x_torch, t_torch):
    return (np.pi**2 - 1) * torch.exp(-t_torch) * torch.sin(np.pi * x_torch)

def total_loss(model, x_bc, y_bc, x_pde,
               w_bc=100.0, w_pde=1.0,
               alpha=1.0, source_fn=None):
    """
    Loss totale = w_bc * L_bc + w_pde * L_pde
    Retourne : (loss_totale, l_bc, l_pde)
    """
    l_bc  = loss_bc(model, x_bc, y_bc)
    l_pde = loss_pde(model, x_pde, alpha=alpha, source_fn=source_fn)
    loss  = w_bc * l_bc + w_pde * l_pde  #shape (5000, 1)
    return loss, l_bc, l_pde

LAYERS  = [2, 50, 50, 50, 50, 1]
LR      = 1e-3
EPOCHS  = 3000
W_BC    = 100.0
W_PDE   = 1.0

# Modèle et optimiseur
model     = MLP(LAYERS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Déplacer les données sur le device
x_bc_d  = x_bc.to(DEVICE)
y_bc_d  = y_bc.to(DEVICE)
x_pde_d = x_pde.to(DEVICE)

# Historique
history = {'loss': [], 'l_bc': [], 'l_pde': []}

# --- Boucle ---
for epoch in range(EPOCHS):
    optimizer.zero_grad()

    loss, l_bc, l_pde = total_loss(model, x_bc_d, y_bc_d, x_pde_d,w_bc=W_BC, w_pde=W_PDE,alpha=1.0, source_fn=source_fn)
    loss.backward()
    optimizer.step()

    history['loss'].append(loss.item())
    history['l_bc'].append(l_bc.item())
    history['l_pde'].append(l_pde.item())

    if epoch % 500 == 0:
        print(f'Epoch {epoch:5d} | Loss: {loss.item():.4e} | BC: {l_bc.item():.4e} | PDE: {l_pde.item():.4e}')