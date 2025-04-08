import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

def visualize_tree_graph(root_node):
    G = nx.DiGraph()
    labels_dict = {}
    def add_nodes(n, parent_id=None):
        node_id = id(n)
        labels_dict[node_id] = f"L{n.level}: {n.label[:15]}..."
        G.add_node(node_id)
        if parent_id is not None:
            G.add_edge(parent_id, node_id)
        if hasattr(n, "children"):
            for child in n.children:
                add_nodes(child, node_id)
    add_nodes(root_node)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=False, node_size=500, node_color="lightblue")
    nx.draw_networkx_labels(G, pos, labels_dict, font_size=8)
    plt.title("Hierarchical Clustering Tree")
    plt.show()

def visualize_clustering(data, labels, centers, method="PCA"):
    if method == "PCA":
        reducer = PCA(n_components=2)
        reduced_data = reducer.fit_transform(data)
    elif method == "t-SNE":
        reducer = TSNE(n_components=2, random_state=42)
        reduced_data = reducer.fit_transform(data)
    else:
        raise ValueError("Method must be 'PCA' or 't-SNE'")
    unique = np.unique(labels)
    plt.figure(figsize=(8,6))
    for lab in unique:
        plt.scatter(reduced_data[labels==lab, 0], reduced_data[labels==lab, 1], label=f"Cluster {lab}")
    centers_2d = reducer.transform(centers)
    plt.scatter(centers_2d[:,0], centers_2d[:,1], marker="X", s=200, c="black", label="Centroids")
    plt.title(f"Clustering Visualization using {method}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("Call visualizer functions from the main process.")
