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
    - request at start(bool) : Début d'un IPU + lorsqu'une volonté de prendre la parole a été interpétée 
    - turn_start_word(float) : Pas utile, à supprimer


Lien utiles :

- Définitions et illustrations IPUs : https://www.researchgate.net/figure/Illustration-of-turn-taking-events-IPU-Interpausal-Unit-Turn-for-speaker-A-and_fig1_359613784