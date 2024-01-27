# SAM

### Objectifs du projet : Multimodal approaches to predict turn-taking in natural conversations

- Manipuler différentes modalités et combiner les informations que chaque modalité apporte
- Explorer différents types d’architectures et fusions (early / late / quels features ?)

### Dataset PACO-CHEESE

- Multi modal corpus (audio + vidéo) en français
- 26 dyades de 15-20min
- Annotations : Segmentation de la parole basée sur les silences (IPUs) / Transcription manuelle alignée sur l’audio
- Lien pour télécharger le dataset : [Dataset](https://amubox.univ-amu.fr/s/gkfA7rZCWGQFqif)

Annotations :
- Colonnes :
    - turn_at_start (bool) : Début d'un IPU
    - turn_after (bool) : Changement de locuteur principal
    - yield at end (bool) : Fin d'un IPU 
    - request at start(bool) : Début d'un IPU lorsqu'une volonté de prendre la parole a été interpétée 
    - turn_start_word(float) : Pas utile, à supprimer

Choix : 
- Les bandes son suivantes ont été supprimées car nous n'avons pas les vidéos correspondantes : JLLJ et JRBG
- Pour AAOR nous n'avons que la bande son pour AA ou le mix des deux, nous avons choisi de l'ignorer pour le moment

Lien utiles :

- Définitions et illustrations IPUs : https://www.researchgate.net/figure/Illustration-of-turn-taking-events-IPU-Interpausal-Unit-Turn-for-speaker-A-and_fig1_359613784

### Fichiers python
- `load_data.py` : Créé le fichier data.csv (shape : (9670, 12)) qui traite les différents fichiers dans le dossier *transcr* du dataset et applique nos choix
- `split_data.py` : Génère tous les exemples d'apprentissagesà partir du fichier data.csv des audios et des vidéos. Chaque fichier audio/vidéo et découpé par IPU. 
- ``utils.py`` : Contient les fonctions utiles pour le projet
- ``dataset.py`` : Contient les classes de Dataset construit à partir des fichiers générés par `split_data.py`. On peut y trouver toutes les fonctions de préprocessing des données.
- ``model_{modalité}.py`` : Contient les classes de modèles utilisés pour le projet pour chaque modalité. On y trouve aussi les modèles d'encoder en plongements utils pour notre tâche.
- ``train_{modalité}.py`` : Contient les fonctions d'entrainement et d'évaluation des modèles pour chaque modalité. Permet d'entrainer les modèles et de les sauvegarder.
- ``mainLateFusion.py`` : Contient le classes, les fonctions d'entrainement et d'évaluation des modèles de late Fusion. Permet d'entrainer les modèles et de les sauvegarder.
- ``mainEarlyFusion.py`` : Contient le classes, les fonctions d'entrainement et d'évaluation des modèles de early Fusion. Permet d'entrainer les modèles et de les sauvegarder.


### Procédure de préparation des données

- Télécharger le dataset et le décompresser dans le dossier *data*. Dans ce dossier, il faut créer un dossier *transcr* et y mettre les fichiers *transcr* du dataset. Idem pour les videos dans le dossier *videos*. Enfin, il faut créer un dossier *audios* et y mettre les audios du dataset. Nous décidons de travailler sur les 1_channels, qu'il faut convertir en *.wav*.
- Lancer le script `load_data.py` pour créer le fichier data.csv
- Lancer le script `split_data.py` pour créer les fichiers audios et vidéos découpés par IPU
- Lancer le script `mainLateFusion.py` pour entrainer et évaluer les modèles de late fusion
- Lancer le script `mainEarlyFusion.py` pour entrainer et évaluer les modèles de early fusion

Les fichiers d'entrainement sont par défaut dans un mode d'évaluation des modèles pré-entrainés.
