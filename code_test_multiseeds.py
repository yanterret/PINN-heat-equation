import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
from PINN import MLP
from Loss import total_loss 
from CI import source_fn
from erreur_finale import u_exact

def points(seed, n_pde=5000, N_CI=500, N_CL=250):
    """Régénère TOUTES les données sous contrôle de la seed."""
    rng = np.random.default_rng(seed)          # générateur LOCAL, pas l'état global

    # Points de collocation
    x = rng.uniform(-1.0, 1.0, size=(n_pde, 1))
    t = rng.uniform(0.0, 1.0, size=(n_pde, 1))
    x_pde = torch.tensor(np.hstack([x, t])).float()
    # points par condition aux limites sur les bords

    # conditions en t=0 
    x_ci      = np.zeros((N_CI, 2))               #shape (N_CI=500, 2)
    x_ci[:,0] = np.linspace(-1, 1, N_CI)   # x compris entre (-1 et 1)
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


def exp(layers, seed, epochs=3000):
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_pde_d, x_bc_d, y_bc_d = points(seed)      
    model = MLP(layers).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, _, _ = total_loss(model, x_bc_d, y_bc_d, x_pde_d,
                                w_bc=100.0, w_pde=1.0, source_fn=source_fn)
        loss.backward()
        optimizer.step()
    
    x_test = np.linspace(-1, 1, 200)
    t_test = np.linspace(0,  1, 200)
    X, T_grid = np.meshgrid(x_test, t_test)
    XT = torch.FloatTensor(np.stack([X.flatten(), T_grid.flatten()], axis=1)).to(DEVICE)
    with torch.no_grad():
        T_pred = model(XT).cpu().numpy().reshape(200, 200)

        T_exact = u_exact(X, T_grid)

        return (np.sqrt(np.mean((T_pred - T_exact)**2))
          / np.sqrt(np.mean(T_exact**2)))



architectures = {
    "4c x 8" : [2,8,8,8,8,1],
    "5c x 8" : [2,8,8,8,8,8,1],
}

SEEDS = list(range(1, 31))

resultats = {}
for nom, layers in architectures.items():
    erreurs = np.array([exp(layers, s) for s in SEEDS])
    resultats[nom] = erreurs
    n_rates = (erreurs > 0.02).sum()

    print(f"{nom}")
    print(f"  moyenne : {erreurs.mean()}   médiane : {np.median(erreurs)}")
    print(f"  std     : {erreurs.std()}   min/max : {erreurs.min()} / {erreurs.max()}")
    print(f"  valeurs triées : {[f'{e:.2%}' for e in sorted(erreurs)]}")
    print(f"  runs ratés (>2%) : {n_rates} / {len(SEEDS)}")