# pyro-npsde

Pyro version of NPSDE

## Installation

Install python packages

```
source venv/Scripts/activate
pip3 install -r requirements.txt
```

## How to use

### Train models

```
python3 src/run.py -h
```

### Dataframe formatting

![dataframe_template.csv](data/dataframe_template.csv)

### Task list formatting

![pyro_template.csv](tasks/pyro_template.csv)

### Analysis (Perturbation, Irreversibility)

```
python3 src/npsde_pyro.py -h
```
