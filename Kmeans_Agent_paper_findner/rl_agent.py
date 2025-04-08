import gym
from gym import spaces
import numpy as np
from clustering import ClusteringSystem
from scipy.spatial.distance import cdist
import os
os.environ["OMP_NUM_THREADS"] = "1"

class ClusteringEnv(gym.Env):
    def __init__(self, data, init_k=3, desired_range=(2,4), max_steps=20, max_clusters=10):
        super(ClusteringEnv, self).__init__()
        self.data = data
        self.init_k = init_k
        self.desired_range = desired_range
        self.max_steps = max_steps
        self.current_step = 0
        self.max_clusters = max_clusters
        self.cs = ClusteringSystem(self.data, init_k=self.init_k)
        state_dim = self.max_clusters * 5 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3, self.max_clusters, self.max_clusters])
        self.prev_silhouette = self.cs.get_state()['silhouette']
    def _get_obs(self):
        return self.cs.get_extended_state(max_clusters=self.max_clusters).astype(np.float32)
    def _apply_action(self, action):
        action_type, target_cluster, merge_cluster = action
        if action_type == 0:
            if self.cs.k < 2:
                return
            if target_cluster < self.cs.k and merge_cluster < self.cs.k and target_cluster != merge_cluster:
                self.cs.merge_clusters(target_cluster, merge_cluster)
            else:
                centers = self.cs.centers
                distances = cdist(centers, centers)
                np.fill_diagonal(distances, np.inf)
                idx = np.unravel_index(np.argmin(distances), distances.shape)
                self.cs.merge_clusters(idx[0], idx[1])
        elif action_type == 1:
            if target_cluster < self.cs.k:
                self.cs.split_cluster(target_cluster)
            else:
                unique, counts = np.unique(self.cs.labels, return_counts=True)
                target = unique[np.argmax(counts)]
                self.cs.split_cluster(target)
        elif action_type == 2:
            pass
    def _compute_reward(self):
        state = self.cs.get_state()
        current_silhouette = state['silhouette']
        delta = current_silhouette - self.prev_silhouette
        self.prev_silhouette = current_silhouette
        return delta
    def step(self, action):
        self.current_step += 1
        self._apply_action(action)
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self.current_step >= self.max_steps
        return obs, reward, done, {}
    def reset(self):
        self.cs = ClusteringSystem(self.data, init_k=self.init_k)
        self.current_step = 0
        self.prev_silhouette = self.cs.get_state()['silhouette']
        return self._get_obs()
    def render(self, mode='human'):
        state = self.cs.get_state()
        print(f"Step: {self.current_step}, Clusters: {state['n_clusters']}, Inertia: {state['inertia']:.2f}, Silhouette: {state['silhouette']:.3f}")

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
    env = ClusteringEnv(data, init_k=4, desired_range=(3,6), max_steps=20, max_clusters=10)
    obs = env.reset()
    total_reward = 0
    for _ in range(env.max_steps):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        if done:
            break
    print("Episode finished. Total Reward:", total_reward)
