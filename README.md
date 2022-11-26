# Knowledge tracing to improve student performance
This repo contains the code for *Exploring deep knowledge tracing to improve student performance*, a final project for the Fall 2022 sequence of CS230. 

**Project authors**: Francois Chesnay (fchesnay), Heidi Kim (hyunsunk), and Cameron Mohne (mohnec1) @stanford.edu. 

## Description of the dataset
Tailoring education to a student's ability level is one of the many valuable things an AI tutor can do. The challenge is to predict whether students are able to answer their next questions correctly. This dataset provides the same sort of information a complete education app would have: that student's historic performance, the performance of other students on the same question, metadata about the question itself, and more.

Dataset is available at: https://www.kaggle.com/competitions/riiid-test-answer-prediction/data

A comprehensive description of the dataset is provided in the article *EdNet: A Large-Scale Hierarchical Dataset in Education* available at at: https://arxiv.org/abs/1912.03072


 ## Metrics
 - Binary AUC



## Setup
Install Conda

Check conda is up to date:
```
$ conda update conda

```


Create and activate a virtual environment in Conda:
```
$ conda create -n knowledge_tracing python=3.8
$ conda activate knowledge_tracing

```

Install the required dependency using PIP3 by running the following command in the root directory of the project:
```
$ pip3 install -r requirements.txt

```

To install the Kaggle library using pip:
```
pip3 install kaggle

```
Download the dataset from Kaggle at https://www.kaggle.com/competitions/riiid-test-answer-prediction/data and install it in the input directory using the following commands:
```
$ cd input
$ kaggle competitions download -c riiid-test-answer-prediction 

```
Unzip riiid-test-answer-prediction.zip and copy in the directory **input** the files lectures.csv, questions.csv and train.csv, as well as the directory **riiid-test-answer-prediction**.


For help on configuring the Kaggle library, please see https://towardsdatascience.com/how-to-search-and-download-data-using-kaggle-api-f815f7b98080


# SAINT-pytorch
A Simple pyTorch implementation of "Towards an Appropriate Query, Key, and Value Computation for Knowledge Tracing" based on https://arxiv.org/abs/2002.07033.  



**SAINT**: Separated Self-AttentIve Neural Knowledge Tracing. SAINT has an encoder-decoder structure where exercise and response embedding sequence separately enter the encoder and the decoder respectively, which allows to stack attention layers multiple times.  

## SAINT model architecture  
<img src="https://github.com/chesnay/cs230_projectv2/arch_from_original_paper.JPG">


## Running the model


or
```
$ conda activate knowledge_tracing
$ python main.py --config_file config/saint.yaml

```



