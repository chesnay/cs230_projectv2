# Knowledge tracing to improve student performance
This repo contains the code for *Exploring deep knowledge tracing to improve student performance*, a final project for the Fall 2023 sequence of CS230. 

**Project authors**: Francois Chesnay (fchesnay), Heidi Kim (hyunsunk), and Cameron Mohne (mohnec1). 

## Setup
Install Conda

Check conda is up to date:
```
conda update conda

```


Create and activate a virtual environment in Conda:
```
conda create -n knowledge_tracing python=3.8
conda create activate knowledge_tracing

```

To run the trained model, run the following in the root directory: 
```
$ pip install -r requirements.txt
$ python main.py --config_file config/akt.yaml

```
