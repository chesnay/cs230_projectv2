# Knowledge tracing to improve student performance
This repo contains the code for *Exploring deep knowledge tracing to improve student performance*, a final project for the Fall 2022 sequence of CS230. 

**Project authors**: Francois Chesnay (fchesnay), Heidi Kim (hyunsunk), and Cameron Mohne (mohnec1) @stanford.edu. 

## Description of the dataset

Tailoring education to a student's ability level is one of the many valuable things an AI tutor can do. Your challenge in this competition is a version of that overall task; you will predict whether students are able to answer their next questions correctly. You'll be provided with the same sorts of information a complete education app would have: that student's historic performance, the performance of other students on the same question, metadata about the question itself, and more.

Dataset is available at: https://www.kaggle.com/competitions/riiid-test-answer-prediction/data



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
$ pip install -r requirements.txt

```

Download the dataset from Kaggle at https://www.kaggle.com/competitions/riiid-test-answer-prediction/data and install it in the input directory:


## Running the models

To run the trained model, run the following in the root directory of the project: 
```
$ conda activate knowledge_tracing
$ python main.py --config_file config/ak.yaml

```
