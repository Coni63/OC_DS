# Explication des fichiers

Pour ce projet, j'ai tet� pas mal de mod�les dont certains ne passait pas en m�moire sur les notebooks. Du coup, ils sont cod� dans le folder mod�les.
De ce fait, ce readme a �t� mis pour expliquer le contenu du dossier


## Dossier dataset

Un 1er dossier dataset a �t� fait pour regrouper les dataset initiaux, les alleger, les fusionner et les pr�parer pour les diff�rents mod�les. N'uploadant pas les dataset, ce dossier ne devrait pas �tre pr�sent sur github.

## Dossier img

Ce dossier regroupe les images utilis�es pour le rapport/la pr�sentation.

## Dossier models

Comme expliqu� precedeement, mon 1er mod�le crashait sur Notebook de ce fait, j'ai d�compos� le code en plusieurs petits scripts avec des fonction particuli�res :

* 1 - ligher : Ce script supprime les features completement inutiles juste pour sauvegardes les 12 dataset dans une version plus l�g�re
* 2 - merger :  Ce script a �t� fait initiallement pour fusionner les 12 dataset, affiner les features necessaire et tester certaines aggr�gations
* 3 - generate_matrices : Ce script a �t� fait pour tout le pr�processing avec les mod�les (scaling, split train/test, OHE). Les matrices �taient ensuite sauvegard� pr�te pour les mod�les
* 4.X - Tests de diff�rents mod�les
* 10 - Ce script reprennait les 12 mois all�gers pour faire un dataset sur 1 ans afin de tester ARIMA avec juste la feature temps/retard
* 11 - Ce dataset a �t� pris sur le site pour les 6 premiers mois de 2017. Ne pouvant pas utliser de split train/test avec ARIMA car il fonctionne "par r�currence", j'ai voulu le tester sur 2017. Par contre ARIMA ne pr�dit que 7j, du coup je comparait la prediction 2016 sur les courbes de 2017 pour voir le fitting
* 20 - Generate_pickle . Ce dataset sauvegarde certaines donn�es dans pickle pour l'impl�mentation de l'API.
* 25 - merger_model_ 2 : Ce script fusionne les 12 mois et pr�pare le dataset pour le mod�le 2 qui a mieux fonctionn�
* 30 - merger_model_ 3 : Ce script fusionne aussi les 12 mois pour un mod�le que je n'ai pas fini �tant "hors sujet"
* log_result :  Ce fichier regroupe les r�sultat des grid search pour le mod�le 1  (ayant de mauvais r�sultat malgr� un MAE assez faible) en fonction d'hyperparam�tres et du format du dataset

## Dossier prod

Ce dossier regroupe les fichier necessaire � l'API

## Autres fichiers :

* ARIMA : ce notebook regroupe les tests fait sur ARIMA avec le dataset 2017 en tester (mod�le un peu hors sujet)
* Exploration : ce notebook regroupe diverses explorations faites et � la fin la visualisation des resultats de certains mod�les
* Merge+Cleanup : ce notebook regroupe le code permettant la pr�paration des dataset mais celui-ci n'est pas � jour suite � labandon avec les crashs
* Model_X : Ces notebooks regroupent les diff�rents models test�s. Pour "Model", il n'est aussi pas � jour car il a �t� fait avec les script pr�sent�s dans le folder "Models".
