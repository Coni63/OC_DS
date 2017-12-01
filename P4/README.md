# Explication des fichiers

Pour ce projet, j'ai teté pas mal de modèles dont certains ne passait pas en mémoire sur les notebooks. Du coup, ils sont codé dans le folder modèles.
De ce fait, ce readme a été mis pour expliquer le contenu du dossier


## Dossier dataset

Un 1er dossier dataset a été fait pour regrouper les dataset initiaux, les alleger, les fusionner et les préparer pour les différents modèles. N'uploadant pas les dataset, ce dossier ne devrait pas être présent sur github.

## Dossier img

Ce dossier regroupe les images utilisées pour le rapport/la présentation.

## Dossier models

Comme expliqué precedeement, mon 1er modèle crashait sur Notebook de ce fait, j'ai décomposé le code en plusieurs petits scripts avec des fonction particulières :

* 1 - ligher : Ce script supprime les features completement inutiles juste pour sauvegardes les 12 dataset dans une version plus légère
* 2 - merger :  Ce script a été fait initiallement pour fusionner les 12 dataset, affiner les features necessaire et tester certaines aggrégations
* 3 - generate_matrices : Ce script a été fait pour tout le préprocessing avec les modèles (scaling, split train/test, OHE). Les matrices étaient ensuite sauvegardé prête pour les modèles
* 4.X - Tests de différents modèles
* 10 - Ce script reprennait les 12 mois allégers pour faire un dataset sur 1 ans afin de tester ARIMA avec juste la feature temps/retard
* 11 - Ce dataset a été pris sur le site pour les 6 premiers mois de 2017. Ne pouvant pas utliser de split train/test avec ARIMA car il fonctionne "par récurrence", j'ai voulu le tester sur 2017. Par contre ARIMA ne prédit que 7j, du coup je comparait la prediction 2016 sur les courbes de 2017 pour voir le fitting
* 20 - Generate_pickle . Ce dataset sauvegarde certaines données dans pickle pour l'implémentation de l'API.
* 25 - merger_model_ 2 : Ce script fusionne les 12 mois et prépare le dataset pour le modèle 2 qui a mieux fonctionné
* 30 - merger_model_ 3 : Ce script fusionne aussi les 12 mois pour un modèle que je n'ai pas fini étant "hors sujet"
* log_result :  Ce fichier regroupe les résultat des grid search pour le modèle 1  (ayant de mauvais résultat malgré un MAE assez faible) en fonction d'hyperparamètres et du format du dataset

## Dossier prod

Ce dossier regroupe les fichier necessaire à l'API

## Autres fichiers :

* ARIMA : ce notebook regroupe les tests fait sur ARIMA avec le dataset 2017 en tester (modèle un peu hors sujet)
* Exploration : ce notebook regroupe diverses explorations faites et à la fin la visualisation des resultats de certains modèles
* Merge+Cleanup : ce notebook regroupe le code permettant la préparation des dataset mais celui-ci n'est pas à jour suite à labandon avec les crashs
* Model_X : Ces notebooks regroupent les différents models testés. Pour "Model", il n'est aussi pas à jour car il a été fait avec les script présentés dans le folder "Models".
