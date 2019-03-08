import sys
import numpy as np
import pandas as pd
from priority_dict import priority_dict

from tree import Tree

class Graph:
    def __init__(self, file_name=None, adj_matrix=None):
        if adj_matrix:
            self.adj_matrix = adj_matrix
            self.num_vertices = self.adj_matrix.shape[0]
        elif file_name:
            df = pd.read_csv(file_name, header=None)

            # Dictionary to store cities' indices
            self.vert_name_2_id = {}
            cnt = 0

            # Iterate all rows and assign an unique id to each city
            for _, row in df.iterrows():
                for city in [row[0], row[1]]:
                    if city not in self.vert_name_2_id:
                        self.vert_name_2_id[city] = cnt
                        cnt += 1
            self.num_vertices = len(self.vert_name_2_id)

            # Create an adjacency matrix of shape (num_vertices, num_vertices)
            self.adj_matrix = np.zeros((self.num_vertices, self.num_vertices))
            # Iterate the rows again to build the adj matrix
            for _, row in df.iterrows():
                idx_1 = self.vert_name_2_id[row[0]]
                idx_2 = self.vert_name_2_id[row[1]]
                # Add the connection to adjacency matrix (both directions)
                self.adj_matrix[idx_1][idx_2] = row[2]
                self.adj_matrix[idx_2][idx_1] = row[2]
            
            self.vert_id_2_name = {v: k for k,v in self.vert_name_2_id.items()}

    def edge_weight(self, u, v):
        return self.adj_matrix[u][v]


def neighbors(G, v):
    """
    Returns indices of v's neighbors
    """
    row = G.adj_matrix[v]
    return np.where(row != 0)[0]


def degree(G, v):
        """
        Returns the degree of a given vertex
        """
        return len(self.neighbors(v))


def dfs_helper(G, current, parent):
    # Scan through the neighbors of the current vertex
    for neighbor in neighbors(G, current):
        # Check if it is already visited or not
        if neighbor not in parent:
            # We visit this vertex from "current", so mark it as the parent
            parent[neighbor] = current
            # Recur!
            dfs_helper(G, neighbor, parent)


def dfs(G, start):
    """
    Perform DFS from the given starting vertex
    """
    # Store parent information
    parent = {start: None}

    # Call the recursive helper function, real work starts here
    dfs_helper(G, start, parent)
    return parent


def components(G):
    """
    Returns the list of connected components in the graph
    """
    big_parent = {}
    component_list = []

    # Perform DFS on every vertices, one completion of a DFS corresponds to one connnected component
    for v in range(G.adj_matrix.shape[0]):
        if v not in big_parent:
            parent = dfs(G, v)
            component_list.append(parent)
            big_parent.update(parent)

    return component_list


def path(G, u, v):
    """
    Returns the path between two vertices if it exists
    """
    # Perform DFS from u
    parent = dfs(G, u)

    # If a path from u to v exists, backtrack the path from vertex 2 using information stored in parent
    if v in parent:
        current = v
        p = []
        while (current is not None):
            p.append(current)
            current = parent[current]
        p.reverse()
        return p
    else:
        return None


def grow_edge(G, current, parent, T):
    for v in neighbors(G, current):
        if v not in parent:
            parent[v] = current
            new_subtree = Tree(G.vert_id_2_name[v], T)
            grow_edge(G, v, parent, new_subtree)


def spanning_tree(G, start):
    """
    Returns a spanning tree of G rooted at start, using DFS
    """
    T = Tree(G.vert_id_2_name[start])
    parent = {start: None}
    grow_edge(G, start, parent, T)

    return T


def prim(G):
    """
    Returns a minimum spanning tree of G using Prim's algorithm
    """
    parent = {0: None}
    C = priority_dict()
    for v in range(G.num_vertices):
        C[v] = sys.maxsize
    C[0] = 0

    while len(C) != 0:
        v = C.pop_smallest()
        for w in neighbors(G, v):
            if (w in C) and (C[w] > G.adj_matrix[v][w]):
                C[w] = G.adj_matrix[v][w]
                parent[w] = v

    tree_nodes = {}
    for v in range(G.num_vertices):
        tree_nodes[v] = Tree(G.vert_id_2_name[v])

    for v in parent:
        if parent[v] is not None:
            tree_nodes[v].set_parent(tree_nodes[parent[v]])
    return tree_nodes[0]


def dijkstra(G, s):
    """
    Find the shortest paths from s to all other vertices in G using Dijkstra's algorithm
    """

    # Priority queue that holds estimated shortest distance
    Q = priority_dict()
    for v in range(G.num_vertices):
        Q[v] = sys.maxsize
    # Initialization
    Q[s] = 0
    parent = {s: None}
    D = {}
    
    while (len(Q) != 0):
        # Extract the vertex with smallest distance to s
        u = Q.smallest()
        D[u] = Q[u]
        Q.pop_smallest()

        # Perform relaxation on neighbors of u
        for v in neighbors(G, u):
            if v not in D:
                alt = D[u] + G.edge_weight(u, v)
                if alt < Q[v]:
                    Q[v] = alt
                    parent[v] = u
    return D, parent


def shortest_path(G, vert_name_1, vert_name_2):
    """
    Returns the shortest path with its distance from u to v
    """
    u = G.vert_name_2_id[vert_name_1]
    v = G.vert_name_2_id[vert_name_2]

    D, parent = dijkstra(G, u)
    current = v
    path = []
    while current is not None:
        path.append(current)
        current = parent[current]
    path = [G.vert_id_2_name[node] for node in path]
    path.reverse()
    return path, D[v]
