# Knowledge tracing to improve student performance
This repo contains the code for *Exploring deep knowledge tracing to improve student performance*, a final project for the Fall 2022 sequence of CS230. 

**Project authors**: Francois Chesnay (fchesnay), Heidi Kim (hyunsunk), and Cameron Mohne (mohnec1) @stanford.edu. 

## Setup
Install Conda

Check conda is up to date:
```
conda update conda

```


Create and activate a virtual environment in Conda:
```
conda create -n knowledge_tracing python=3.8
conda activate knowledge_tracing

```

Install the required dependency using PIP3 by running the following command in the root directory of the project:
```
$ pip install -r requirements.txt

```

To run the trained model, run the following in the root directory of the project: 
```
$ python main.py --config_file config/ak.yaml

```
