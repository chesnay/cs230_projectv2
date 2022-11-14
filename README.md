# Knowledge tracing to improve student performance
This repo contains the code for *Exploring deep knowledge tracing to improve student performance*, a final project for the Fall 2023 sequence of CS230. 

**Project authors**: Francois Chesnay (fchesnay), Heidi Kim (hyunsunk), and Cameron Mohne (mohnec1). 

## Setup
Install Conda

Check conda is up to date:
```
conda update conda

```

Install pip
```
sudo apt install python3-pip

```


create a virtual environment in Conda, for example:
```
conda create -n yourenvname python=3.8

```

To run the trained model, run the following in the root directory: 
```
$ pip install -r requirements.txt
$ python main.py --config_file config/akt.yaml

```
