import networkx as nx
import numpy as np
import random as rann

def graph_model():

    # Create a directed graph
    G = nx.DiGraph()

    # Define the edges and distances
    edges = np.array([
        [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [5, 4], [5, 6], [6, 5],
        [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10],
        [11, 12], [12, 13], [13, 12], [13, 14], [14, 13], [14, 15], [15, 14], [15, 16], [16, 15],
        [16, 9], [9, 16], [1, 7], [8, 2], [2, 6], [3, 7], [5, 3], [6, 4], [6, 12], [10, 5],
        [11, 7], [7, 9], [10, 8], [9, 15], [10, 16], [14, 10], [11, 15], [13, 11], [12, 14],
        [10, 6], [2, 7], [7, 2], [3, 6], [6, 3], [7, 10], [10, 7], [6, 11], [11, 6], [10, 15],
        [15, 10], [11, 14], [14, 11], [12, 11]
    ])

    distances = 5 * np.array([
        20, 50, 40, 20, 50, 30, 40, 50, 20, 50, 50, 40, 20, 50, 30, 40, 30, 40, 20, 50, 30, 20,
        30, 20, 50, 30, 40, 30, 20, 50, 40, 50, 20, 50, 50, 30, 50, 40, 50, 50, 20, 50, 40, 20,
        50, 30, 40, 30, 50, 30, 20, 50, 40, 50, 20, 50, 30, 50, 30, 40, 50, 50
    ])
    #print(len(edges))


    # Add edges and weights to the graph
    for i in range(len(edges)):
        G.add_edge(edges[i,0], edges[i,1], weight=distances[i])

    for idx, edge in enumerate(G.edges()):
        G.edges[edge]['index'] = idx
    # Get the edge weights
    #distances = nx.get_edge_attributes(G, 'weight')
    # Get the number of edges
    nedge = G.number_of_edges()
    #print(distances)

    # Plot the graph
    #pos = nx.spring_layout(G)
    #nx.draw(G, pos, with_labels=True)
    #nx.draw_networkx_edges(G, pos, edge_color='gray')
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=distances)
    #plt.show()

    ## very importent
    edgematrix=np.array(G.edges)
    #print(edgematrix)
    return edgematrix,nedge,G,distances