# On crée les points à l'intérieur du domaine
# j'ai défini sur numpy  mais c est faisable aussi avec des fonctions pytorch

N_PDE = 5000

# à remplacer avec random pour les tests reproductibles
# rng = np.random.default_rng(seed)

x = np.random.uniform(-1.0, 1.0, size=(N_PDE, 1)) #shape (N_PDE ,1)
t = np.random.uniform(0, 1.0, size=(N_PDE, 1))    #shape (N_PDE ,1)
x_pde_np = np.hstack([x,t])      #shape (N_PDE ,2) float64
x_pde = torch.tensor(x_pde_np).float()    #torch.Size([5000, 2]) torch.float32

N_CI  = 500   # points condition initiale source
N_CL  = 250   # points par condition aux limites sur les bords

# conditions en t=0 
x_ci      = np.zeros((N_CI, 2))               #shape (N_CI=500, 2)
x_ci[:,0] = np.linspace(-1, 1, N_CI)   # x compris entre (-1 et 1)
x_ci[:,1] = 0.0                        # t=0
y_ci      = np.sin(np.pi * x_ci[:,0:1])   #shape (500, 1)

# conditions en t=-1
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