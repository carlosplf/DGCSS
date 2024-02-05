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
usage: run.py [-h] [--planetoid] [--bench] [--epochs EPOCHS] [--n_graphs N_GRAPHS] [--level LEVEL]

options:
  -h, --help           show this help message and exit
  --planetoid          Use a Planetoid dataset.
  --bench              Use a Benchmark clustering graph dataset.
  --epochs EPOCHS      Define number of EPOCHS for training.
  --n_graphs N_GRAPHS  Define number of Graphs to run.
  --level LEVEL        Define the difificulty level from the Dataset. Options: 'easy_small', 'hard_small', 'easy', 'hard'.
```

### Authors:

Carlos Pereira Lopes Filho

Guilherme Henrique Messias
