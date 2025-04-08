# paper_finder.py
import pandas as pd
import numpy as np
import requests

OLLAMA_API_URL = "http://localhost:11435/api/generate"
MODEL_NAME = "gemma3"

def choose_cluster_label(paper_text, candidate_labels):
    prompt = (
        f"给定论文摘要：\n{paper_text}\n\n"
        "请从下面的候选聚类标签中选择一个最符合论文主题的标签：\n" +
        "\n".join([f"{i+1}. {label}" for i, label in enumerate(candidate_labels)]) +
        "\n\n只回复标签文本，不要任何附加内容。"
    )
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        chosen_label = result.get("completion", "").strip()
    except Exception as e:
        chosen_label = f"API调用失败: {str(e)}"
    return chosen_label

def find_paper(paper_text, clusters_info_df):
    max_level = clusters_info_df["level"].max()
    selected_labels = []
    current_level = 0
    
    # 遍历每一层的候选簇
    while current_level <= max_level:
        df_level = clusters_info_df[clusters_info_df["level"] == current_level]
        candidate_labels = df_level["label"].tolist()
        if not candidate_labels:
            break
        chosen_label = choose_cluster_label(paper_text, candidate_labels)
        print(f"Level {current_level} 选择的标签: {chosen_label}")
        selected_labels.append(chosen_label)
        # 筛选下一层中以选定标签为父标签的候选簇
        # 假设 clusters_info_df 中有父标签字段 "parent_label"
        df_next = clusters_info_df[(clusters_info_df["level"] == current_level + 1) &
                                   (clusters_info_df["parent_label"] == chosen_label)]
        # 如果没有，则当前就是最终层
        if df_next.empty:
            break
        clusters_info_df = df_next  # 更新候选簇为下一层
        current_level += 1
    
    return selected_labels

def main():
    clusters_info_df = pd.read_csv("clusters_info.csv")
    print("加载的聚类信息（前5行）：")
    print(clusters_info_df.head())
    
    proceed_dir = "processed_texts\\a-note-on-the-evaluation-of-generative-models_ICLR_2016.txt"
    paper_text = open(proceed_dir, "r", encoding="utf-8").read()
    
    selected_labels = find_paper(paper_text, clusters_info_df)
    print("最终选择的标签路径：", " -> ".join(selected_labels))
    
if __name__ == "__main__":
    main()
