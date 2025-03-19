# MultimodalMutationsPDAC
This repository include the implementation of the framework introduced by the work entitled "A Multimodal Framework for Assessing the Link between Pathomics, Transcriptomics, and Pancreatic Cancer Mutations".
I
f you find this repository useful for your research, please cite our paper in *Computerized Medical Imaging and Graphics*:
```
@article{BERLOCO2025102526,
title = {A multimodal framework for assessing the link between pathomics, transcriptomics, and pancreatic cancer mutations},
journal = {Computerized Medical Imaging and Graphics},
volume = {123},
pages = {102526},
year = {2025},
issn = {0895-6111},
doi = {https://doi.org/10.1016/j.compmedimag.2025.102526},
url = {https://www.sciencedirect.com/science/article/pii/S0895611125000357},
author = {Francesco Berloco and Gian Maria Zaccaria and Nicola Altini and Simona Colucci and Vitoantonio Bevilacqua},
keywords = {Multimodal Analysis, Pathomics, Transcriptomics, Pancreatic Cancer, Explainability},
}
```


The code is written in both R and Python. 
It is structured as follows:
1. _CLAM_, containing the code of "Clustering-constrained Attention Multiple Instance Learning" [CLAM](https://github.com/mahmoodlab/CLAM) framework, edited for the imaging pipeline (feature extraction and classification). 
2. _Transcriptomic_analysis_, containing the scripts for transcriptomic data analysis.
3. _Multimodal_analysis_, containing the code related to the model ensemble, as well as metrics computation.
4. _NeuralMultimodal_, containing the code of MANN model training and inference.

## Installation and Use
### CLAM
You can clone CLAM from https://github.com/mahmoodlab/CLAM and follow the related instruction for download the foundation models for feature extraction and CLAM training.

If you use CONCH as feature extractor, change the "embed_dim" to 512 parameter in models/model_clam.py (default value 1024). 

NOTE: the code was slight edited for returning the attention scores from the attention net as features during the inference, along with the predicted probabilities.

*training_setup_commands.py* contains a list of preset prompt commands for data splitting, training, eval and heatmaps creation. 

### Trascriptomic Analysis

This directory contains several sub-directories for transcriptomic data analysis:
1. *AE*, scripts for training scripts for training and validating the Autoencoders
2. *R*, with a .R file for DeSeq2 analysis
3. *voting classifiers*, for training and validating RF, XGBoost (train_voting) and MLP (train_voting_Keras), including XAI modules.

### Multimodal Analysis

As stated in the [paper](https://authors.elsevier.com/a/1kncL3BesszcYb), multimodal analysis was performed with a voting and a MANN approach.
Given the probabilities predicted by unimodal models, *Multimodal Analysis* folder contains the script for combining the results of CLAM and transcriptomic models using a voting-based approach.

Specifically, 
1. *clam_ensemble.py* takes a list of experiments of CLAM in a format "main_folder/namefolder_FeatureExtraction/experiment_folder", where "experiment_folder" should have the gene name and clam size, divided by "_" as last two words.
2. *trascriptomic_ensemble.py* takes a list of folders (named as the target gene),which contains the probabilities in a csv file (named as the trained model).
The output of clam_ensemble.py and trascriptomic_ensemble.py should be saved in the same folder. 
3. *eval_ensemble.py* takes as input the output of clam_ensemble.py and trascriptomic_ensemble.py (saved in the same folder), computes the ensamble and returns the metrics. *metrics4paper.py* reformat the output of eval_ensemble.py

*NeuralMultimodal* folder contains the code for running the experiments for multimodal model (traing and validation). The dataset should be put in a folder in a format:
data/train(or test)/genes/features where features are the csv containing CLAM attention scores and AE/DEG transcriptomic features.




