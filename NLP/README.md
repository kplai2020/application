## NLP : 

The NLP section aims to demo how a ML/DL solution is presented as a small ML/DL project. 

This repository is structured as presented following:

- /data # shared by a source
    - /train.tsv # train dataset
    - /test.tsv # test dataset
- /models
    - /model.py # main class of the model
    - /train.py # script to train the model
    - /test.py # script to run model on test set
    - /pipeline.py # a pipeline to use the trained model to detect entities(NER).
        1. Input: str a sentence
        2. Output: a list of entities if you are doing NER
    - /config.py # parameters setting
    - /data_prep.py # all data related preparation
- [/biobert_v1.1_pubmed](https://drive.google.com/drive/folders/1b4N2DFNLZomkYTyPZOWZ98SGbF5P90mZ?usp=sharing) # biomedical pre-trained models 
- [/results](https://drive.google.com/drive/folders/1U27eyYw2Luh0-3-XruqgI0cSUBzxfac8?usp=sharing) # trained and produced from this test
    - /ner_result.csv # stored all the records that are detected by the pipeline.py
    - /trained_model.pt # trained model params that are learnt by the train.py


NOTE: 
1. The files of this project are shared through 2 channels: Github and Google Drive.
2. The epoch is intentionally designed to be set as 1 for checking purposes and hence, to gain a more promising result do re-traine the model.
3. The dataset is only designed to read on 5000 train records for demo.
4. The ner_result.csv file is just for a quick reference and hence, to gain a more promising result do re-traine the model.
5. For your quick start, you may want to start with pipeline.py, the script is generally run as stated above.
