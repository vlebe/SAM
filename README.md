# SAM

# SAM 2023 Project

This project focuses on evaluating the performance of various multimodal fusion approaches for predicting turn-taking in dialogue.

## Introduction

In this project, we aim to evaluate the performance of different multimodal fusion approaches in the context of predicting turn-taking in dialogue. The main objective is to explore and compare the results obtained by various methods, with a particular emphasis on integrating multiple modalities into the prediction process.

## Data Processing

Our dataset consists of videos with corresponding audios, along with a CSV file containing information for each training example. Each example represents an Inter-Personal Unit (IPU) and includes information such as the speaker, start and end times, text, and an indication of whether there is a turn-taking after the IPU.

We performed data preprocessing by segmenting audio and video files for each IPU. Additionally, we extracted four frames from each video IPU and filtered audio clips shorter than 0.5 ms. This preprocessing resulted in approximately 10,000 examples, which we further downsampled to balance the dataset to 4,000 examples.

## Late Fusion

We began by studying the performance of a late fusion model for our task. The objective was to build an architecture with a classifier for each modality and then combine the results to improve the final prediction quality.

### Audio Model

We initially employed MFCC (Mel-Frequency Cepstral Coefficients) to represent audio features and experimented with various architectures, including RNNs and MLPs. We observed that the models learned to predict only 0s due to the imbalanced dataset, leading us to introduce class weights in the loss function. We tested different hyperparameters and architectures but found limited improvement.

### Video Model

For video classification, we utilized a pre-trained ResNet to extract features from each frame and then applied a GRU to capture temporal dynamics. Despite various adjustments, including changing the depth of MLPs, the results were not satisfactory, especially with imbalanced loss.

### Text Model

We employed DistilCamemBERT, a lightweight version of CamemBERT, to generate text embeddings. We added an MLP layer on top of the [CLS] token output for classification. Results were similar to audio and video models, with challenges in handling imbalanced data.

### Fusion and Results

We combined predictions from the audio, video, and text models using two methods: linear combination with learnable weights and majority voting. While the late fusion model demonstrated better accuracy and precision in most cases, the voting ensemble approach showed promising results, especially with balanced loss.

## Early Fusion

We explored another fusion method, early fusion, aiming to integrate features from different modalities before classification to create a unified representation.

### Feature Representation

We extracted features for each modality, including audio MFCCs, video ResNet embeddings, and text DistilCamemBERT embeddings.

### Fusion

We combined these features by averaging them and passing them through an MLP for classification.

### Results

Early fusion results were mixed, with challenges in outperforming unimodal models. Further exploration and refinement of the early fusion architecture are needed to improve performance.

## Conclusion

In conclusion, this project allowed us to evaluate various modalities for predicting turn-taking in dialogue. While each model could be improved, fusion of modalities showed potential in enhancing system performance, particularly in the late fusion approach. Further analysis and optimization of fusion methods are essential for advancing multimodal dialogue prediction systems.

