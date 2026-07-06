DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LAYERS  = [2, 32, 32, 32, 1]
LR      = 1e-3
EPOCHS  = 3000
RESAMPLE_FREQ = 500

# Modèle et optimiseur

model= MLP(LAYERS).to(DEVICE)
optimizer= torch.optim.Adam(model.parameters(), lr=LR)

# Déplacer les données sur le device

x_bc_d  = x_bc.to(DEVICE)
y_bc_d  = y_bc.to(DEVICE)
x_pde_d = x_pde.to(DEVICE)

history = {'loss': [], 'l_bc': [], 'l_pde': []}

for epoch in range(EPOCHS):
    
    if epoch % RESAMPLE_FREQ == 0 and epoch > 0:

        optimizer.zero_grad()
        loss, l_bc, l_pde = total_loss(model, x_bc_d, y_bc_d, x_pde_d,w_bc=w_bc, w_pde=w_pde,alpha=1.0, source_fn=source_fn)
        loss.backward()
        optimizer.step()

        history['loss'].append(loss.item())
        history['l_bc'].append(l_bc.item())
        history['l_pde'].append(l_pde.item())

        if epoch % 500 == 0:
            print(f'Epoch {epoch:5d} | Loss: {loss.item():.4e} | BC: {l_bc.item():.4e} | PDE: {l_pde.item():.4e}')