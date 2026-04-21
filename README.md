# Projet Machine Learning - Obesity Dataset

Ce projet transforme un script Jupyter en code Python modulaire pour classifier le jeu de données `Obesity_Dataset.arff` avec un **RandomForestClassifier**.

## Parcours complet : Atelier 1 a Atelier 4

### Atelier 1 : Script Jupyter fonctionnel

Objectif : obtenir un premier pipeline machine learning qui fonctionne de bout en bout dans un notebook.

Realise dans le projet :

- preparation des donnees du dataset Obesity,
- entrainement initial du modele,
- evaluation des performances,
- verification des metriques obtenues.

Resultat : base de travail validee pour passer a la modularisation.

### Atelier 2 : Modularisation du code

Objectif : transformer le notebook en code Python structure avec des fonctions reutilisables.

Realise dans le projet :

- creation de [model_pipeline.py](model_pipeline.py) avec :
  - prepare_data()
  - train_model()
  - evaluate_model()
  - save_model()
  - load_model()
- creation de [main.py](main.py) pour executer train/evaluate en ligne de commande.

Resultat : pipeline ML modulaire, maintenable et reutilisable.

### Atelier 3 : Automatisation avec Makefile

Objectif : automatiser l'execution du projet et les etapes CI.

Realise dans le projet :

- creation de [Makefile](Makefile) avec les cibles : install, train, evaluate, api, format, lint, security, test, ci,
- creation de [requirements.txt](requirements.txt),
- ajout de [test_model_pipeline.py](test_model_pipeline.py) pour valider le pipeline.

Resultat : execution simplifiee avec des commandes standardisees.

### Atelier 4 : Exposition REST avec FastAPI

Objectif : exposer la prediction en service REST.

Realise dans le projet :

- creation de [app.py](app.py),
- endpoint `POST /predict` pour la prediction,
- endpoint `POST /retrain` pour le reentrainement (excellence),
- endpoint `GET /example-request` pour une requete d'exemple,
- acces a la documentation interactive via /docs.

Resultat : API REST operationnelle pour integration dans d'autres applications.

## Modèle utilisé

- Modèle : `RandomForestClassifier`
- Paramètres principaux : `n_estimators=200`, `class_weight="balanced"`, `random_state=42`

## Structure du pipeline

- `prepare_data()` : charge le fichier ARFF, prépare les données et fait la séparation train/test.
- `train_model()` : entraîne le modèle Random Forest.
- `evaluate_model()` : calcule l’accuracy, la matrice de confusion et le rapport de classification.
- `save_model()` : sauvegarde le modèle entraîné avec joblib.
- `load_model()` : recharge le modèle sauvegardé.

## Fichiers principaux

- [model_pipeline.py](model_pipeline.py) : fonctions du pipeline ML.
- [main.py](main.py) : point d’entrée CLI pour exécuter le projet.
- [Makefile](Makefile) : automatisation des tâches.

## Exécution

```bash
python main.py --mode train --data-path "archive (1)/Obesity_Dataset.arff" --model-path "artifacts/obesity_model.joblib"
python main.py --mode evaluate --data-path "archive (1)/Obesity_Dataset.arff" --model-path "artifacts/obesity_model.joblib"
```

## Résultat obtenu

Lors des tests du pipeline, le modèle a donné les performances suivantes :

- Accuracy : `0.860248447204969`
- Matrice de confusion :

```text
[
  [8, 5, 1, 1],
  [0, 125, 7, 0],
  [0, 14, 97, 7],
  [0, 0, 10, 47]
]
```

### Rapport de classification

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 1 | 1.00 | 0.53 | 0.70 | 15 |
| 2 | 0.87 | 0.95 | 0.91 | 132 |
| 3 | 0.84 | 0.82 | 0.83 | 118 |
| 4 | 0.85 | 0.82 | 0.84 | 57 |

### Résumé global

- Accuracy : `0.8602`
- Macro avg F1-score : `0.8183`
- Weighted avg F1-score : `0.8574`

## Atelier 3 : Makefile

Le projet inclut un [Makefile](Makefile) pour automatiser l’installation, l’exécution du modèle et les vérifications CI.

### Commandes utiles

```bash
make install
make train
make evaluate
make format
make lint
make security
make test
make ci
```

### Validation

Les cibles suivantes ont été validées avec succès :

- `make test`
- `make lint`
- `make security`

Le scan de sécurité est limité aux fichiers applicatifs pour éviter les faux positifs liés aux tests ou aux dépendances.

## Atelier 4 : API REST avec FastAPI

Le projet inclut [app.py](app.py), qui expose le modèle sous forme de service REST.

### Introduction

Ce tutoriel guide la creation d'un service REST pour realiser des predictions avec la fonction predict() d'un modele machine learning. L'objectif est de transformer le modele en API REST utilisable par d'autres applications.

### I. Etape 01 : Preparer l'environnement

1. Installer FastAPI et Uvicorn.

- Dans WSL, activer l'environnement virtuel :

```bash
source venv/bin/activate
```

- Installer les dependances API :

```bash
pip install fastapi uvicorn
```

2. Ajouter les dependances dans [requirements.txt](requirements.txt).

- fastapi
- uvicorn

### Ce qui a ete fait exactement dans ce projet

- Les dependances fastapi et uvicorn sont ajoutees dans [requirements.txt](requirements.txt).
- L'installation des dependances a ete executee dans l'environnement du projet via `pip install -r requirements.txt`.
- Le fichier [app.py](app.py) expose deja la fonction predict via la route `POST /predict`.
- Le projet inclut aussi l'endpoint d'excellence `POST /retrain`.
- Le lancement API est automatise par la cible `make api` dans [Makefile](Makefile).

### II. Etape 02 : Assurer que le modele est deja entraine et sauvegarde

Avant d'utiliser la route `POST /predict`, il faut disposer d'un fichier modele sauvegarde avec la fonction `save_model()`.

Dans ce projet, le fichier attendu est :

- [artifacts/obesity_model.joblib](artifacts/obesity_model.joblib)

Si le fichier n'existe pas encore, entrainer puis sauvegarder le modele avec :

```bash
make train
```

ou avec la commande Python directe :

```bash
python main.py --mode train --data-path "archive (1)/Obesity_Dataset.arff" --model-path "artifacts/obesity_model.joblib"
```

Remarque : dans certains tutoriels, le nom du fichier peut etre `model.joblib`. Dans ce projet, le nom retenu est `obesity_model.joblib`.

### II. Etape 02 : Creer le fichier app.py

1. Creer le fichier [app.py](app.py).

2. Ajouter le code necessaire pour :

- definir une route HTTP POST pour effectuer des predictions,
- utiliser le modele sauvegarde pour predire la sortie,
- retourner le resultat de la prediction.

Ce projet contient deja [app.py](app.py) avec :

- `POST /predict` pour la prediction,
- chargement du modele depuis [artifacts/obesity_model.joblib](artifacts/obesity_model.joblib),
- reponse JSON contenant la classe predite.

### III. Etape 03 : Tester l'API localement

1. Demarrer le serveur FastAPI avec Uvicorn :

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Cette commande active le rechargement automatique (`--reload`) et expose l'API sur toutes les interfaces reseau (`--host 0.0.0.0`) au port 8000.

Dans ce projet, vous pouvez aussi utiliser :

```bash
make api
```

2. Acceder a la documentation interactive :

- Ouvrir un navigateur puis acceder a : http://127.0.0.1:8000/docs
- Cette adresse est indicative. Selon votre environnement, adaptez l'adresse machine et le port utilises.

Liens de test pour ce projet :

- Local : http://127.0.0.1:8000/docs
- Reseau local (exemple machine actuelle) : http://192.168.1.8:8000/docs

Statut de verification :

- Verification HTTP effectuee : code 200 sur /docs

3. Tester votre prediction en exposant le service REST avec FastAPI.

- Verifier que l'endpoint `POST /predict` est fonctionnel depuis Swagger (`/docs`) ou avec une requete REST.
- Une reponse valide retourne un JSON de type :

```json
{
  "predicted_class": 2
}
```

### Excellence : affichage web du resultat selon les criteres saisis

Une solution web a ete ajoutee dans ce projet pour saisir les criteres et afficher le resultat de prediction.

- URL de l'interface web : http://127.0.0.1:8000/web
- Cette page consomme le service `POST /predict` et affiche directement la classe predite.
- Technologie utilisee : interface HTML/JavaScript servie par FastAPI (solution web "autres").

### Captures de resultat (3 captures)

#### Capture 1 - Documentation Swagger

Capture validee : page principale Swagger UI accessible via `http://127.0.0.1:8000/docs`.

Elements visibles :

- titre `Obesity Model API`,
- endpoints `GET /`, `POST /predict`, `POST /retrain`, `GET /example-request`,
- schemas de reponse (`PredictionRequest`, `PredictionResponse`, `RetrainRequest`, `RetrainResponse`).

Legende suggeree :
Capture 1 - Documentation interactive FastAPI chargee avec succes.

#### Capture 2 - Endpoint /retrain

Capture validee : panneau `POST /retrain` ouvert dans Swagger.

Elements visibles :

- `Request body` JSON avec `data_path`, `model_path`, `test_size`, `random_state`,
- reponses `200 Successful Response` et `422 Validation Error`.

Legende suggeree :
Capture 2 - Verification de l'endpoint `/retrain` et de son schema d'entree/sortie.

#### Capture 3 - Endpoint /predict

Capture validee : panneau `POST /predict` ouvert dans Swagger.

Elements visibles :

- schema du JSON d'entree (criteres utilisateur),
- reponse `200` avec `predicted_class`,
- section des erreurs de validation.

Legende suggeree :
Capture 3 - Verification de l'endpoint `/predict` pour la prediction de classe.

### Routes disponibles

- `GET /` : test simple de disponibilité.
- `POST /predict` : prédiction de la classe d’obésité.
- `POST /retrain` : réentraînement du modèle et sauvegarde sur disque.
- `GET /example-request` : exemple de commande REST pour `/predict`.

### Lancer l’API

```bash
make api
```

L'API ecoute sur `http://0.0.0.0:8000`.
Pour tester depuis la meme machine, utilisez generalement `http://127.0.0.1:8000`.

### Exemple de requête REST (prédiction)

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"Sex\":2,\"Age\":21,\"Height\":170,\"Overweight_Obese_Family\":2,\"Consumption_of_Fast_Food\":2,\"Frequency_of_Consuming_Vegetables\":3,\"Number_of_Main_Meals_Daily\":2,\"Food_Intake_Between_Meals\":2,\"Smoking\":2,\"Liquid_Intake_Daily\":2,\"Calculation_of_Calorie_Intake\":2,\"Physical_Excercise\":3,\"Schedule_Dedicated_to_Technology\":3,\"Type_of_Transportation_Used\":4}"
```

### Capture de test (resultat attendu)

Reponse typique de `POST /predict` :

```json
{
  "predicted_class": 2
}
```

Reponse typique de `POST /retrain` :

```json
{
  "message": "Model retrained and saved successfully.",
  "model_path": "E:/AliBotheiana/artifacts/obesity_model.joblib",
  "metrics": {
    "accuracy": 0.860248447204969
  }
}
```

### Excellence : retrain exposé en REST

La fonctionnalité de réentraînement est exposée via `POST /retrain`, ce qui permet de mettre à jour le modèle sans modifier le code source.

## Synthèse complète des travaux réalisés

### Atelier 2 : Modularisation du notebook en code Python structuré

Le script initial a été transformé en architecture modulaire avec :

- `prepare_data()` pour le chargement et le prétraitement des données ARFF.
- `train_model()` pour l'entraînement du modèle.
- `evaluate_model()` pour l'évaluation des performances.
- `save_model()` et `load_model()` pour la persistance du modèle.

Fichiers livrés :

- [model_pipeline.py](model_pipeline.py)
- [main.py](main.py)

### Atelier 3 : Automatisation avec Makefile

Un `Makefile` fonctionnel a été ajouté pour automatiser :

- l'installation (`make install`),
- l'entraînement (`make train`),
- l'évaluation (`make evaluate`),
- la qualité et CI (`make format`, `make lint`, `make security`, `make test`, `make ci`),
- le lancement API (`make api`).

Fichiers livrés :

- [Makefile](Makefile)
- [requirements.txt](requirements.txt)
- [test_model_pipeline.py](test_model_pipeline.py)

### Atelier 4 : Exposition REST avec FastAPI

Une API REST fonctionnelle a été développée pour exposer la prédiction et le réentraînement :

- `POST /predict` : prédire la classe d'obésité.
- `POST /retrain` : réentraîner et sauvegarder le modèle (excellence).
- `GET /example-request` : obtenir une requête d'exemple.

Fichier livré :

- [app.py](app.py)

### Atelier 5 : MLflow avec backend SQL

Le suivi d'experimentation utilise maintenant SQLite via [mlflow.db](mlflow.db), et les artefacts sont stockes dans [mlartifacts](mlartifacts) afin de garder le backend de tracking propre et compatible avec l'onglet Overview de MLflow.

Fichier de configuration principal :

- [mlflow_utils.py](mlflow_utils.py)

## Validation finale

Les validations suivantes ont été exécutées avec succès :

- exécution du pipeline en mode train,
- exécution du pipeline en mode evaluate,
- test unitaire : `1 passed`,
- lint : `All checks passed`,
- sécurité (Bandit) sur le code applicatif : `No issues identified`,
- pipeline CI complet via `make ci` : succès.

## Résultat modèle retenu

- Accuracy : `0.860248447204969`
- Macro avg F1-score : `0.8183`
- Weighted avg F1-score : `0.8574`

## Atelier 5 : Introduction a MLflow

### Objectifs

- Comprendre le role de MLflow pour suivre les executions et gerer les modeles.
- Configurer MLflow pour ce projet.

### Integration realisee dans le projet

- Suivi des experiences avec `mlflow.log_params()` et `mlflow.log_metrics()`.
- Enregistrement du modele avec `mlflow.sklearn.log_model()`.
- Configuration locale du tracking dans [mlflow.db](mlflow.db) avec artefacts dans [mlartifacts](mlartifacts) via [mlflow_utils.py](mlflow_utils.py).
- Journalisation automatique pendant :
  - l'entrainement CLI dans [main.py](main.py),
  - le reentrainement API dans [app.py](app.py) (`POST /retrain`).

### Commandes MLflow

1. Lancer un entrainement pour generer un run :

```bash
make train
```

2. Ouvrir l'interface MLflow :

```bash
make mlflow-ui
```

3. Consulter l'UI MLflow dans le navigateur :

- http://127.0.0.1:5000

### Livrable Atelier 5

Le projet est integre avec MLflow : suivi des parametres/metriques, stockage des runs et enregistrement des modeles dans l'interface MLflow.

## Seance 6 : Conteneurisation avec Docker

### Objectifs

- Conteneuriser l'API FastAPI pour un deploiement reproductible.
- Lancer l'API et MLflow dans des conteneurs.

### Fichiers Docker ajoutes

- [Dockerfile](Dockerfile)
- [docker-compose.yml](docker-compose.yml)
- [.dockerignore](.dockerignore)

### Commandes Docker (image seule)

```bash
make docker-build
make docker-run
```

Tester l'API conteneurisee :

- http://127.0.0.1:8000/docs

Arreter le conteneur :

```bash
make docker-stop
```

### Commandes Docker Compose (API + MLflow)

```bash
make docker-compose-up
```

Liens de test :

- API Swagger : http://127.0.0.1:8000/docs
- MLflow UI : http://127.0.0.1:5000

Arret des services :

```bash
make docker-compose-down
```

## Seance 7 : Supervision continue

### Objectifs

- Verifier en continu la disponibilite de l'API et de MLflow.
- Controler les compteurs de supervision dans la base SQL MLflow.
- Disposer d'un rapport rapide pour diagnostiquer les erreurs "No data available".

### Livrables

- [monitoring_report.py](monitoring_report.py)
- Cibles Makefile `monitor` et `monitor-strict`
- Interface de test moderne : [app.py](app.py) via `/interface` et `/web`

### Commandes

1. Lancer les services :

```bash
make api
make mlflow-ui
```

2. Executer le rapport de supervision :

```bash
make monitor
```

3. Executer la supervision stricte avec alertes bloquantes :

```bash
make monitor-strict
```

4. Generer des donnees reelles pour les graphes MLflow Overview :

```bash
make traffic
```

Pour plus de volume (demo/capture ecran) :

```bash
make traffic-heavy
```

### Interface de test moderne

Ouvrir l'interface principale du projet dans le navigateur :

- http://127.0.0.1:8000/interface

Cette interface permet de :

- saisir les criteres du modele,
- lancer une prediction en direct,
- voir le resultat immediatement,
- consulter des statistiques live du projet.

Le rapport affiche :

- statut HTTP de `http://127.0.0.1:8000/docs`,
- statut HTTP de `http://127.0.0.1:5000`,
- compteurs MLflow (`runs`, `metrics`, `datasets`, `inputs`, `evaluation_datasets`, `trace_info`).
- alertes automatiques si les seuils minimums ne sont pas atteints.

### Resultat attendu

- `api.ok = true`
- `mlflow.ok = true`
- `mlflow_store.exists = true`

Si `evaluation_datasets` ou `trace_info` valent 0, il faut relancer au moins une prediction (`POST /predict`) ou un entrainement (`make train`).

En mode strict, la commande echoue si :

- l'API ou MLflow ne repond pas,
- la base MLflow est absente,
- `trace_info` est inferieur au seuil attendu,
- `evaluation_datasets` est inferieur au seuil attendu.


