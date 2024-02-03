from plotly import graph_objs as go
import matplotlib.pyplot as plt
import networkx as nx


# Função de plot
def show_graph(G):
    # Graph Connections
    edge_x = []
    edge_y = []

    # adicionando as coordenadas
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    # definindo cor e estilo das arestas
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Nodes
    node_x = []
    node_y = []

    # adicionando as coordenadas
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    # definindo cor e estilo dos vértices
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
           size=25,
           line_width=2))

    node_labels = []
    for node in G.nodes():
        node_labels.append(G.nodes[node]['label'])

    node_trace.marker.color = node_labels

    # visualizando!
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False),
                    yaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False))
                    )
    fig.show()


def plot_weights(G, features):
    weights = [edata['weight'] for u, v, edata in G.edges(data=True)]

    pos = nx.spring_layout(G, seed=42)
    # Crie um mapa de cores com base nos pesos das arestas
    edge_color_map = plt.cm.get_cmap('Reds')  # Escolha o colormap desejado
    edge_colors = [edge_color_map(weight) for weight in weights]

    # Dicionario de plot dos vertices
    # dict_color = {0: 'red', 1: 'blue', 2: 'green'}

    label_color_mapping = {
        0: 'red',
        1: 'blue',
        2: 'green',
        3: 'yellow',
        4: 'black'
    }

    node_colors = []
    for node_features in features:
        node_color = node_features.tolist().index(1)
        node_colors.append(label_color_mapping[node_color])

    # Desenhe o grafo com cores representando os pesos das arestas
    nx.draw(G, with_labels=True,
            edge_color=edge_colors, edge_cmap=edge_color_map,
            node_color=node_colors,
            font_weight='bold', arrows=False, width=3.0, pos=pos)

    # Adicione uma barra de cores para representar os valores
    sm = plt.cm.ScalarMappable(cmap=edge_color_map,
                               norm=plt.Normalize(vmin=min(weights),
                                                  vmax=max(weights)))
    sm._A = []  # hack para evitar o bug do matplotlib
    plt.colorbar(sm, label='Pesos das Arestas', ax=plt.gca())

    plt.show()
