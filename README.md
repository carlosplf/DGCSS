# GAT Experiments

:construction: **UNDER CONSTRUCTION!** :construction:

Analyzing how the Attention mechanism in GAT networks can be used to optimize algorithms for processing, clustering, and analysis in Graphs.

### How to run

#### Runners:

This projects contains two different runners, one with the GAT model itself, and another one with the GAE encoder and decoder arround the GAT model.

#### Datasets:

Using the cli arguments, you can chose between the Cora Planetoid dataset and a Dummy small graph dataset.

**NOTE:** the GAT runner is not ready yet to run with the Dummy dataset, just with Cora.

#### Arguments:

```
usage: run.py [-h] [--dummy] [--cora] [--gae] [--epochs EPOCHS]

options:
  -h, --help       show this help message and exit
  --dummy          Use a dummy smal graph for testing.
  --cora           Use the Cora Planetoid dataset.
  --gae            Use GAE encoder and decoder.
  --epochs EPOCHS  Define number of EPOCHS for training.
```

### Authors:

Carlos Pereira Lopes Filho

Guilherme Henrique Messias
