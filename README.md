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
- `load_data.py` : Créé le fichier data.csv (shape : (13861, 12)) qui traite les différents fichiers dans le dossier *transcr* du dataset et applique nos choix
- `split_data.py` : Génère tous les exemples d'apprentissagesà partir du fichier data.csv des audios et des vidéos. Chaque fichier audio/vidéo et découpé par IPU. 