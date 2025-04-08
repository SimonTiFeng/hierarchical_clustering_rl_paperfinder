import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict
import torch

class HierarchicalClustering:
    def __init__(self, data, documents, max_levels=3, min_samples=10, desired_range=(2, 4), max_clusters=10):
        self.data = data
        self.documents = documents
        self.max_levels = max_levels
        self.min_samples = min_samples
        self.desired_range = desired_range
        self.max_clusters = max_clusters
        self.cluster_id_counter = 0
        self.cluster_tree = defaultdict(list)
        self.cluster_labels = {}

    def choose_best_k(self, data):
        best_k = 2
        best_score = -1
        for k in range(self.desired_range[0], min(self.desired_range[1] + 1, len(data))):
            kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            if score > best_score:
                best_k = k
                best_score = score
        return best_k

    def cluster_recursive(self, data, level, parent_label):
        if len(data) < self.min_samples or level > self.max_levels:
            return
        k = self.choose_best_k(data)
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
        labels = kmeans.fit_predict(data)
        for i in range(k):
            cluster_data = data[labels == i]
            cluster_label = f"{parent_label}/{i}"
            self.cluster_tree[parent_label].append((cluster_label, cluster_data))
            self.cluster_labels[cluster_label] = cluster_data
            self.cluster_recursive(cluster_data, level + 1, cluster_label)

    def run(self):
        self.cluster_recursive(self.data, 1, "root")
        return self.cluster_tree, self.cluster_labels

    def save_cluster_centers(self, output_path="clusters.csv"):
        records = []
        for label, data in self.cluster_labels.items():
            level = label.count("/")
            center = np.mean(data, axis=0)
            size = len(data)
            records.append({"level": level, "label": label, "center": center.tolist(), "size": size})
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)

if __name__ == "__main__":
    embeddings = np.load("embeddings.npy")
    with open("filenames.txt", "r", encoding="utf-8") as f:
        filenames = [line.strip() for line in f.readlines()]
    hc = HierarchicalClustering(embeddings, filenames)
    tree, labels = hc.run()
    hc.save_cluster_centers()
