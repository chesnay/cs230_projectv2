# Knowledge tracing to improve student performance
This repo contains the code for *Exploring deep knowledge tracing to improve student performance*, a final project for the Fall 2022 sequence of CS230. 

**Project authors**: Francois Chesnay (fchesnay), Heidi Kim (hyunsunk), and Cameron Mohne (mohnec1) @stanford.edu. 

## Description of the dataset
Tailoring education to a student's ability level is one of the many valuable things an AI tutor can do. The challenge is to predict whether students are able to answer their next questions correctly. This dataset provides the same sort of information a complete education app would have: that student's historic performance, the performance of other students on the same question, metadata about the question itself, and more.

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
$ pip3 install -r requirements.txt

```

To install the Kaggle library using pip:
```
pip install kaggle

```
Download the dataset from Kaggle at https://www.kaggle.com/competitions/riiid-test-answer-prediction/data and install it in the input directory using the following commands:
```
$ cd input
$ kaggle competitions download -c riiid-test-answer-prediction 

```
For help on configuring the Kaggle library, please see https://towardsdatascience.com/how-to-search-and-download-data-using-kaggle-api-f815f7b98080


## Running the models

To run the trained model, run the following in the root directory of the project: 
```
$ conda activate knowledge_tracing
$ python main.py --config_file config/akt.yaml

```
