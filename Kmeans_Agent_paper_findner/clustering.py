import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

class ClusteringSystem:
    def __init__(self, data, init_k=3, random_state=42):
        self.data = data
        self.random_state = random_state
        self.k = init_k
        self.labels = None
        self.centers = None
        self.inertia = None
        self.silhouette = None
        self._run_kmeans()
    def _run_kmeans(self):
        kmeans = KMeans(n_clusters=self.k, random_state=self.random_state, n_init="auto")
        self.labels = kmeans.fit_predict(self.data)
        self.centers = kmeans.cluster_centers_
        self.inertia = kmeans.inertia_
        if self.k > 1:
            self.silhouette = silhouette_score(self.data, self.labels)
        else:
            self.silhouette = -1
    def update_clusters(self, new_labels, new_centers):
        self.labels = new_labels
        self.centers = new_centers
        self.inertia = np.sum([np.sum((self.data[self.labels == i] - self.centers[i])**2) for i in range(len(self.centers))])
        self.k = len(self.centers)
        if self.k > 1:
            self.silhouette = silhouette_score(self.data, self.labels)
        else:
            self.silhouette = -1
    def merge_clusters(self, cluster_idx1, cluster_idx2):
        mask1 = (self.labels == cluster_idx1)
        mask2 = (self.labels == cluster_idx2)
        new_cluster_data = self.data[mask1 | mask2]
        new_center = np.mean(new_cluster_data, axis=0)
        new_centers = []
        new_labels = np.copy(self.labels)
        new_label = 0
        mapping = {}
        for i in range(self.k):
            if i == cluster_idx1 or i == cluster_idx2:
                continue
            mapping[i] = new_label
            new_centers.append(self.centers[i])
            new_label += 1
        mapping[cluster_idx1] = new_label
        mapping[cluster_idx2] = new_label
        new_centers.append(new_center)
        for i in range(len(new_labels)):
            new_labels[i] = mapping[new_labels[i]]
        self.update_clusters(new_labels, np.array(new_centers))
        return self.labels, self.centers
    def split_cluster(self, cluster_idx):
        mask = (self.labels == cluster_idx)
        cluster_data = self.data[mask]
        if len(cluster_data) < 2:
            print("Not enough samples to split")
            return self.labels, self.centers
        kmeans = KMeans(n_clusters=2, random_state=self.random_state, n_init="auto")
        local_labels = kmeans.fit_predict(cluster_data)
        local_centers = kmeans.cluster_centers_
        new_centers = []
        new_labels = np.copy(self.labels)
        new_label_mapping = {}
        new_label = 0
        for i in range(self.k):
            if i != cluster_idx:
                new_centers.append(self.centers[i])
                new_label_mapping[i] = new_label
                new_label += 1
        new_centers.append(local_centers[0])
        new_centers.append(local_centers[1])
        new_label_split1 = new_label
        new_label_split2 = new_label + 1
        idx = 0
        for i in range(len(new_labels)):
            if new_labels[i] == cluster_idx:
                if local_labels[idx] == 0:
                    new_labels[i] = new_label_split1
                else:
                    new_labels[i] = new_label_split2
                idx += 1
            else:
                new_labels[i] = new_label_mapping[new_labels[i]]
        self.update_clusters(new_labels, np.array(new_centers))
        return self.labels, self.centers
    def get_state(self):
        return {"n_clusters": self.k, "inertia": self.inertia, "silhouette": self.silhouette}
    def get_extended_state(self, max_clusters=10):
        n_samples = self.data.shape[0]
        global_center = np.mean(self.data, axis=0)
        features = []
        for i in range(max_clusters):
            if i < self.k:
                idx = np.where(self.labels == i)[0]
                cluster_data = self.data[idx]
                cluster_size = len(cluster_data) / n_samples
                center = self.centers[i]
                center_2d = center[:2]
                distances = np.linalg.norm(cluster_data - center, axis=1)
                mean_distance = np.mean(distances)
                center_global_distance = np.linalg.norm(center - global_center)
                features.extend([cluster_size] + center_2d.tolist() + [mean_distance, center_global_distance])
            else:
                features.extend([0, 0, 0, 0, 0])
        global_state = [self.inertia, self.silhouette]
        return np.array(features + global_state)
    def get_reduced_data(self, method="PCA"):
        if method == "PCA":
            reducer = PCA(n_components=2)
        elif method == "t-SNE":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=self.random_state)
        else:
            raise ValueError("Choose 'PCA' or 't-SNE'")
        return reducer.fit_transform(self.data)

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
    cs = ClusteringSystem(data, init_k=4)
    print(cs.get_state())
    print(cs.get_extended_state())
