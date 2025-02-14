# DGCSD (Deep Graph Clustering with Seed Detection)

## Overview

In this study, we propose **DGCSD**, a novel approach designed to push the boundaries of state-of-the-art performance in node clustering within complex networks. Traditional deep clustering methods often rely on external clustering algorithms to identify representative elements, but these methods can be hampered by the influence of initially formed groups and typically focus only on the content information of each example, neglecting the rich topological structure of the data.

DGCSD addresses these limitations with a three-module design:

- **Embedding Module:** Utilizes a graph attentional network to capture topological information from the data.
- **Seed Selection Module:** Detects representative nodes (seeds) in the graph, ensuring robust initial groupings.
- **Self-Supervised Module:** Leverages the detected representative nodes to guide the clustering process and refine results.

Our goal is to establish DGCSD as a state-of-the-art method for deep graph clustering by effectively integrating both content and topological information to achieve superior clustering performance.


## Features

- **Enhanced Clustering Accuracy:** Mitigates the limitations of external clustering algorithms by directly integrating seed detection.
- **Topological Awareness:** Incorporates graph attentional networks to capture essential structural information.
- **Self-Supervision:** Uses detected seeds to iteratively refine clustering results.
- **Modular Architecture:** Simplifies customization and extension of the algorithm for various applications.

## Installation

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/DGCSD.git
   cd DGCSD
   pip install -r requirements.txt
   python run.py --help


## Authors:

Alan Demétrius Baria Valejo

Carlos Pereira Lopes Filho

Guilherme Henrique Messias
