LAYERS  = [2, 32, 32, 32, 1]
LR      = 1e-3
EPOCHS  = 3000
W_BC    = 100.0
W_PDE   = 1.0

# Modèle et optimiseur nouveaux blocs qui définissent cosine (scheduler) et lbfgs

model= MLP(LAYERS).to(DEVICE)

optimizer= torch.optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = EPOCHS,eta_min = 1e-5)

optimizer_lbfgs = torch.optim.LBFGS(model.parameters(),lr=0.1,max_iter=20, line_search_fn='strong_wolfe' )# max iter nombre d essais par step(), line_search_fn test pour avoir le bon lr 

# recalcul la backprop pour que lbfgs choississe la bonne valeur

def closure():
    optimizer_lbfgs.zero_grad()
    loss, _, _ = total_loss(model, x_bc_d, y_bc_d, x_pde_d, source_fn=source_fn)
    loss.backward()
    return loss

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
    scheduler.step()
    
    history['loss'].append(loss.item())
    history['l_bc'].append(l_bc.item())
    history['l_pde'].append(l_pde.item())

    if epoch % 500 == 0:
        print(f'Epoch {epoch:5d} | Loss: {loss.item():.4e} | BC: {l_bc.item():.4e} | PDE: {l_pde.item():.4e}')
        
for epoch in range(100):

    optimizer_lbfgs.step(closure)

    loss, l_bc, l_pde = total_loss(model, x_bc_d, y_bc_d, x_pde_d, source_fn=source_fn)
    
    history['loss'].append(loss.item())
    history['l_bc'].append(l_bc.item())
    history['l_pde'].append(l_pde.item())
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch:5d} | Loss: {loss.item():.4e} | BC: {l_bc.item():.4e} | PDE: {l_pde.item():.4e}')
