## Lancer le code

Prérequis : Python ≥ 3.10.

```bash
git clone https://github.com/yanterret/PINN-heat-equation.git
cd PINN-heat-equation
pip install -r documentation/requirement.txt

cd PINN_heat_equation_1D
python code_entrainement_modele.py    # comparaison des architectures
python train_adam.py                  # Adam seul en 3c*32
python train_adam_cosine_LBFGS.py     # Adam + CosineAnnealingLR + L-BFGS
python ../methode-odeint.py           # odeint
```

Les résultats sont écrits au format CSV dans `PINN_heat_equation_1D/resultats/`.