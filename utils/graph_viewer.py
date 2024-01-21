from plotly import graph_objs as go


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
