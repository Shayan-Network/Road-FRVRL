import networkx as nx
import numpy as np
import random as rann

def get_path(edge_matrix, A, B, numbredeges):
    G = nx.DiGraph()
    
    # Generate random edge weights
    distances = 100 * [rann.uniform(0, 1) for _ in range(numbredeges)]
    
    # Add edges to the graph
    for i in range(numbredeges):
        G.add_edge(edge_matrix[i][0], edge_matrix[i][1], weight=distances[i])
    
    # Try to find the shortest path from A to B
    try:
        path = nx.dijkstra_path(G, A, B)
    except nx.NetworkXNoPath:
        # If no path is found, return an empty list
        path = []
    
    return path
