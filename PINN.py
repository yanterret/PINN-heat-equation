import torch.nn as nn


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

                # tanh en fonction d activation partout sauf à la fin  
                net.append(nn.Tanh())   

        # on crée le réseau couches par couches en déballant la liste
        self.net = nn.Sequential(*net)

        # Initialisation avec égalisation des variance en back et foward prop 
        # pour éviter le gradient vanishing  (Xavier)
        for m in self.net:   
             
            # on prends les sorties de chaque couches en déballant 
            # les couches et fonctions d'activations du réseau
            # ex : si tranh() = False si Linear = True (sorte de dtype)    
            if isinstance(m, nn.Linear):  
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)
