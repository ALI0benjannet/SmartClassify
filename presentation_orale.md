b # Version courte pour présentation orale

Ce projet utilise le jeu de données `Obesity_Dataset.arff` pour prédire la classe d’obésité avec un **RandomForestClassifier**.

## Ce que j’ai fait

1. J’ai chargé et préparé les données.
2. J’ai séparé les données en apprentissage et en test.
3. J’ai entraîné un modèle Random Forest.
4. J’ai évalué les performances du modèle.
5. J’ai sauvegardé le modèle avec `joblib`.
6. J’ai créé un `Makefile` pour automatiser les tâches.

## Modèle utilisé

- Modèle : `RandomForestClassifier`
- Paramètres principaux :
  - `n_estimators=200`
  - `class_weight="balanced"`
  - `random_state=42`

## Résultat obtenu

- Accuracy : `0.8602`
- Macro avg F1-score : `0.8183`
- Weighted avg F1-score : `0.8574`

## Interprétation

Le modèle donne de bons résultats globaux. Il classe correctement la majorité des observations et reste adapté à ce type de problème de classification.

## Conclusion

Ce travail montre comment transformer un notebook Jupyter en projet Python modulaire, testable et automatisable avec un `Makefile`.