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
python3 src/run.py parallel-run pyro data/data1.csv data/metadata1.csv tasks/tasklist1.csv 5 --report log.txt
```

### Dataframe formatting

![dataframe_template.csv](data/dataframe_template.csv)
![metadata_template.json](data/metadata_template.json)

### Task list formatting

![pyro_template.csv](tasks/pyro_template.csv)

### Analysis (Perturbation, Irreversibility)

```
python3 src/npsde_pyro.py -h
python3 src/npsde_pyro.py data/model1.pt data/data1.csv data/metadata1.csv --graph graph1.png --irreversibility --imputation 50 50 --perturbation series1.csv series2.csv
```
