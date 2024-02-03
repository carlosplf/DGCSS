from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


def download_dataset(dataset_name="PubMed"):
    # Defina o diretório onde você deseja armazenar o conjunto de dados
    root = './data/'

    # Baixe o conjunto de dados e o carregue
    dataset = Planetoid(root=root, name=dataset_name,
                        transform=T.NormalizeFeatures(), pre_transform=None)

    # Imprima algumas informações sobre o conjunto de dados
    print('Número de gráficos (grafos):', len(dataset))
    print('Número de classes:', dataset.num_classes)
    print('Número de recursos:', dataset.num_node_features)

    return dataset
