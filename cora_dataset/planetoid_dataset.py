from torch_geometric.datasets import Planetoid


def download_dataset():
    # Defina o diretório onde você deseja armazenar o conjunto de dados
    root = './data/Cora'

    # Baixe o conjunto de dados Cora e o carregue
    dataset = Planetoid(root=root, name='Cora', transform=None, pre_transform=None)

    # Imprima algumas informações sobre o conjunto de dados
    print('Número de gráficos (grafos):', len(dataset))
    print('Número de classes:', dataset.num_classes)
    print('Número de recursos:', dataset.num_node_features)

    return dataset[0]