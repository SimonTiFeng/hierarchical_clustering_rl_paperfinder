import os
import numpy as np
from transformers import BertTokenizer, BertModel
from rl_agent import ClusteringEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from sklearn.feature_extraction.text import TfidfVectorizer

def read_processed_text(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                documents.append(f.read().strip())
    return documents

def extract_bert_features(documents):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    embeddings = []
    for doc in documents:
        inputs = tokenizer(doc, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        doc_emb = outputs.last_hidden_state[:, 0, :].detach().numpy()
        embeddings.append(doc_emb)
    return np.array(embeddings).squeeze()

def main():
    processed_dir = "processed_texts"
    documents = read_processed_text(processed_dir)
    print(f"Read {len(documents)} documents.")
    use_bert = True
    if use_bert:
        features = extract_bert_features(documents)
    else:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(documents)
        features = tfidf_matrix.toarray()
    
    env = ClusteringEnv(features, init_k=3, desired_range=(2,4), max_steps=20, max_clusters=10)
    callback = CheckpointCallback(save_freq=5000, save_path="./models/", name_prefix="ppo_clustering")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, callback=callback)
    model.save("ppo_clustering_final")
    print("Model saved as 'ppo_clustering_final.zip'")
    
    obs = env.reset()
    total_reward = 0
    for _ in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        if done:
            break
    print("Episode finished. Total Reward:", total_reward)

if __name__ == "__main__":
    main()
