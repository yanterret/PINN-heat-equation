# PINN — Équation de la chaleur 1D

Résolution de l'équation de la chaleur 1D par un réseau de neurones informé par la physique (Physics-Informed Neural Network), à partir du code tutoriel de la FAU.

### Mes contributions
En m'appuyant sur ce code de référence, j'ai acquis le squelette d'un PINN, puis mené trois études :

- **Choix de l'architecture optimale** (nombre de couches et de neurones) pour ce problème
- **Influence de l'initialisation Xavier** sur la convergence
- **Comparaison des optimiseurs** Adam seul vs Adam + CosineAnnealingLR + L-BFGS


## Mes résultats clés

### Comparaison des architectures

**Objectif :** déterminer la meilleure architecture (nombre de neurones et de couches) pour résoudre ce problème.

**Protocole :** on prendra 30 seeds qui fixent les points de collocation dans le domaine, afin de mesurer les performances indépendamment d'un maillage particulier. Optimiseur Adam, avec Xavier.

Un réseau est noté i c × j, avec i le nombre de couches cachées et j le nombre de neurones par couche. Le nombre de paramètres est calculé en sommant, sur chaque couche, la taille de la matrice des poids et le nombre de biais :

$dimension_{entrée} × dimension_{sortie} +$ Nombre de neurones

On distinguera 2 interprétations de résultats :

* La **médiane** peut être interprétée comme un indicateur pour savoir si, globalement, **le modèle trouve la meilleure solution**.
* L'**écart-type** (et le nombre de runs ratés) nous indique si le modèle renvoie des résultats proches de la moyenne en produisant le moins possible de « runs ratés » ; on pourra appeler cela la **stabilité du modèle**.

| Architecture | Nombre de paramètres | Médiane | Écart-type | Runs (sur 30) avec > 2 % d'erreur |
|:---:|:---:|:---:|:---:|:---:|
| 5c × 64 | 17 153 | 0,47 % | $\color{red}{\text{2,41\%}}$ | 10
| 5c × 50 | 10 501 | 0,42 % | 0,90 % | 7
| 5c × 40 | 6 761 | 0,39 % | $\color{red}{\text{1,65\%}}$ | 6
| 5c × 32 | 4 353 | 0,47 % | $\color{red}{\text{1,31\%}}$ | 2
| 5c × 16 | 1 153 | 0,55 % | 0,59 % | 2
| 5c × 8 | 321 | $\color{red}{\text{0,80\%}}$ | $\color{green}{\text{0,32\%}}$ | 0
| 4c × 64 | 12 993 | 0,38 % | $\color{red}{\text{1,76\%}}$ | 11
| 4c × 50 (FAU) | 7 951 | 0,35 % | 0,91 % | 2
| 4c × 40 | 5 121 | 0,51 % | $\color{red}{\text{1,87\%}}$ | 5
| $\color{green}{\text{4c × 32}}$ | 3 297 | 0,33 % | 0,48 % | 1
| 4c × 16 | 865 | 0,52 % | $\color{green}{\text{0,34\%}}$ | 0
| 4c × 8 | 249 | $\color{red}{\text{0,98\%}}$ | $\color{green}{\text{0,40\%}}$ | 2
| 3c × 64 | 8 833 | 0,31 % | 0,92 % | 1
| 3c × 50 | 5 401 | 0,37 % | 0,69 % | 1
| $\color{green}{\text{3c × 40}}$ | 3 481 | 0,37 % | 0,47 % | 1
| 3c × 32 | 2 241 | 0,41 % | 0,94 % | 4
| 3c × 16 | 593 | 0,54 % | $\color{green}{\text{0,36\%}}$ | 1
| 3c × 8 | 161 | $\color{red}{\text{1,12\%}}$ | 0,91 % | 3

Dans ce cadre-là, on remarque que globalement (hors modèles de moins de 1000 paramètres, sujets à l'underfitting) **les modèles donnent un résultat satisfaisant, avec une médiane d'erreur inférieure à 0,6 %**. Ce qui les différencie, c'est leur stabilité.

Grâce à cela, on remarque que le modèle **PINN de FAU a clairement des concurrents pour moins de temps de calcul, tels que le 4c × 32 ou le 3c × 40**, qui convergent certes vers un résultat quasiment identique (médianes à 0,37, 0,33 et 0,35 %) mais en étant plus sûrs de le faire (écarts-types de 0,47 et 0,48 % au lieu de 0,91 %).

On remarque aussi que **le modèle optimal est obtenu avec un nombre de neurones intermédiaire par couche en forme de U**.

Modèles avec un nombre de paramètres **inférieur à 600 (underfitting)** :
* 3c × 8 (161) : médiane 1,12 %
* 4c × 8 (249) : médiane 0,98 %
* 5c × 8 (321) : médiane 0,80 %

Modèles avec un nombre de paramètres **supérieur à 1000, instables** :
* 4c × 64 (12 993) : 11 ratés
* 5c × 64 (17 153) : 10 ratés
* 5c × 50 (10 501) : 7 ratés

De plus, **la profondeur du réseau semble aggraver l'instabilité**, indépendamment du nombre de paramètres. À nombre de paramètres comparable (~7 000), le 3c × 64 reste stable (1 raté) tandis que le 5c × 40 se dégrade (6 ratés).

### Utilité de l'initialisation Xavier

La première chose qui m'a interpellé dans le code de la FAU a été l'initialisation Xavier, que je n'avais pas l'habitude de voir. Il s'agit d'une initialisation des poids codée ainsi :

*nn.init.xavier_normal_(m.weight)*
*nn.init.zeros_(m.bias)*

Pour cela, j'ai reproduit le même protocole, mais avec 10 seeds différentes, sur le modèle initial FAU 4×50 :

| Xavier | Médiane | Écart-type |
|:---:|:---:|:---:|
| Oui | 0,80 % | 1,47 %
| Non | 0,37 % | 1,66 %

Xavier n'a pas d'influence ici : il sert surtout à faire décroître l'écart-type. En effet, il permet d'éviter le *gradient vanishing* dans les réseaux profonds ; or, sur un problème physique simple qui ne nécessite pas un gros réseau, ce n'est pas le facteur limitant. Le solveur sera donc un sujet plus important à traiter. Xavier est donc retiré par la suite, pour économiser du temps de calcul (aussi minime soit-il).

### Test de deux solveurs

Maintenant que l'on a vu que l'initialisation des poids a peu d'importance,j'ai pris le modéle (3c × 32) pour réduire le temps de calcul et testé la diférence de performance — en résultat et en temps de calcul — entre le solveur **Adam** et **Adam + CosineAnnealingLR + L-BFGS**.

Pour la 2ᵉ méthode, j'ai divisé mes 3000 epochs d'entraînement en 2 parties :

* l'une où Adam fonctionne avec un calcul de pas adaptatif du solveur (méthode CosineAnnealingLR) pour plus de performance (2900 epochs)

avec CosineAnnealingLR qui recalcule le pas du solveur Adam avec la formule :
```math
\text{lr(epoch)} = \eta_{min} + 0.5 \,(lr_{initial} - \eta_{min})\left(1 + \cos\left(\pi \cdot \frac{epoch}{T_{max}}\right)\right)
```

Elle sert à avoir un grand learning rate au début pour trouver la zone où se situe le minimum, puis à le réduire pour entrer plus facilement dans la cuve. J'ai posé $T_{max}$ égal au nombre d'epochs et $\eta_{min}$ égal à $10^{-5}$, une valeur standard pour ce solveur.

* l'autre où le solveur L-BFGS termine l'entraînement (100 epochs)

en faisant 20 essais avec ce même solveur pour trouver le chemin optimal.

Sur 10 seeds, on obtient :

|| Adam seul | Adam + CosineAnnealingLR + L-BFGS |
|:---:|:---:|:---:|
| Médiane | 0,52 % | 0,02 %
| Écart-type | 0,68 % | 0,01 %
| Runs au-delà de 2 % d'erreur | 1 | 0
| Temps moyen | 14 s | 64,2 s dont 49 s de L-BFGS

On remarque que CosineAnnealingLR permet de tomber à chaque fois dans la bonne cuve du minimum, avec un écart-type de 0,01 %, et que L-BFGS minimise bien la fonction de coût dans cette cuve. En revanche, le temps est quasiment multiplié par 4, ce qui peut être handicapant si le modèle doit résoudre un problème en temps réel. Finalement, on obtient **×4,58 de temps pour ×22 de précision** : ce sera le rôle de celui qui conçoit le modèle de réduire le nombre d'essais dans le solveur pour trouver un compromis.

## Le problème physique

Ce code permet de simuler la diffusion de la chaleur, représentée par la température T, dans un barreau à une dimension x et dans le temps t, qui respecte l'équation :

```math
\partial_{t}T - \partial^{2}_{xx}T = f(x,t)\quad(1), \quad x\in(-1,1),\quad t\in(0,1)
```

avec T(x,t), le champ de température qui respecte les conditions aux limites (2) :

```math
\left\{\begin{array}{l} \text{dans l'espace} \quad T(-1,t)=T(1,t)=0 \\ \text{dans le temps} \quad T(x,0)=\sin(\pi x) \end{array}\right.
```

Dans notre cas, on prendra $f(x,t) = (\pi^{2} - 1) e^{-t}\sin(\pi x)$ pour illustrer de manière optimale les performances du modèle.
On retrouve cette expression en prenant la solution exacte définie par $T(x,t) = e^{-t}\sin(\pi x)$, qui correspond à la solution de l'équation de la chaleur pour $(x,t)\in(-1,1)\times(0,1)$, puis en l'injectant dans (1).

## Méthode (réseau, activation, coûts)

Pour résoudre cette équation, on aura besoin d'un maillage et d'un réseau de neurones qui prend :
* en entrée, un vecteur (x,t) de dimension ($N_{pde}$,2) que l'on fixe : ce sera notre maillage
* en sortie, un vecteur température T(x,t) de dimension ($N_{pde}$,1), obtenu par entraînement, censé approximer la température en chaque point du maillage tout en respectant les conditions aux limites et l'équation de la chaleur.

($N_{pde}$ faisant référence au nombre de points de collocation qui doivent respecter l'équation de diffusion de la chaleur.)

Pour ce faire, un réseau est constitué de la manière suivante, par neurone :

```math
a^{k+1}_{j} = g\!\left(\sum_{i} w^{k}_{ji}\, a^{k}_{i} + b^{k}_{j}\right)
```

avec :
* $k$ : le numéro de la couche
* $j$ : le neurone dans la couche $k+1$
* $i$ : indice des neurones de la couche $k$
* $a^{k}_{i}$ : la sortie du neurone $i$ de la couche $k$
* $w^{k}_{ji}$ : le poids reliant le neurone $i$ (couche $k$) au neurone $j$ (couche $k+1$)
* $b^{k}_{j}$ : le biais du neurone $j$
* $g$ : la fonction d'activation

##### Le choix de la fonction d'activation

Pour choisir celle-ci, on se base sur 2 critères :
* le fait d'être dérivable deux fois (il faut vérifier l'équation de la chaleur)
* avoir la plage de valeurs la plus large possible pour la dérivée première, afin de capturer le maximum de variations pour la dérivée seconde

Ainsi, le premier critère élimine ReLU, qui donne une dérivée seconde nulle en permanence, et le second nous impose de choisir tanh plutôt que la sigmoïde.

##### Le choix des paramètres

Ainsi, par backpropagation, on optimise les coefficients  $`\theta = (w^{k}_{ij}, b^{k}_{j})`$ du réseau pour que la sortie $y_{\theta}(x,t)$, qui dépend de ces mêmes paramètres, donne $y_{\theta}(x,t) \approx T(x,t)$.

Cette optimisation se fait par la minimisation des fonctions de coût suivantes, avec $d_{p}^{pde}$ qui correspond à $(x_{p},t_{p})$, où $p\in(0,N_{pde})$, qui constituent l'entrée du réseau, donc notre maillage fixe à l'intérieur du domaine :

$$Loss_{pde}(\theta) = \frac{1}{N_{pde}}\sum_{p\in(0,N_{pde})} (\partial_{t}y_{\theta}[d_{p}^{pde}] - \partial^{2}_{xx}y_{\theta}[d_{p}^{pde}]  - f[d_{p}^{pde}] )^{2} $$

qui est une MSE pour que les points de collocation approximent le plus fidèlement possible (1).

Pour l'extérieur, on prendra de la même manière $d_{p}^{CI}$, le maillage aux bords, avec $(x_{p},t_{p})$, où $p\in(0,N_{CI})$ ; on notera $y_{CI}$ les points des conditions initiales traduisant (2) :

$$Loss_{CI}(\theta) = \frac{1}{N_{CI}}\sum_{p\in(0,N_{CI})} (y_{\theta}[d_{p}^{CI}] - y_{CI} )^{2} $$

qui est une MSE pour que les points aux bords approximent le plus fidèlement possible (2).

Avec cela, on construit la loss finale avec les poids associés à chacune des Loss, que l'on notera W :

$$Loss = W_{pde}\ Loss_{pde} +W_{CI}\ Loss_{CI}$$

Finalement, on testera le résultat du code en posant une norme (L2) sur la différence entre ce que le modèle prédit et la solution exacte de l'équation.

## Construction du code

Pour plus de lisibilité, j'ai découpé le code en plusieurs sections :
* création du réseau de neurones
* création des points à l'intérieur du maillage (pde) et aux bords (CI)
* création des fonctions Loss
* boucle d'entraînement
* test du modèle

## Critique de la méthode par PINN

On nuancera l'efficacité de cette méthode de résolution avec, par exemple, la **méthode odeint** vue dans l'UE MINI aux Arts et Métiers, qui servait déjà de méthode de référence. Dans ce cadre-là, on observe une erreur de 7,59·10⁻⁵ % en 14 lignes de code et moins d'une seconde de calcul.

Ce résultat est bien inférieur à celui de n'importe quel réseau entraîné : sur des problèmes simples, les méthodes traditionnelles (différences finies, discrétisations, odeint..) sont plus performantes. Les PINN sont eux plus intéressants sur des problèmes de dimension plus élevée ou lorsqu'on veut résoudre le problème inverse (on a le champ de température et on veut obtenir les constantes de l'équation de la chaleur).

Ce repo sert avant tout de tutoriel pour s'emparer des techniques de résolution par PINN pour ensuite s'attaquer à des problémes ou il est pertinent de l'utiliser.

## Crédits et licence

Ce projet s'appuie sur le code tutoriel PINN 1D du laboratoire
*FAU Chair for Dynamics, Control, Machine Learning and Numerics - Alexander von Humboldt Professorship*
disponible sous licence MIT :
https://github.com/DCN-FAU-AvH/pinns_heat