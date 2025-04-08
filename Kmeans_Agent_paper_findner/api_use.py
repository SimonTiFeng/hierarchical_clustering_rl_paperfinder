import requests
import json

OLLAMA_API_URL = "http://localhost:11435/api/generate"
MODEL_NAME = "gemma3"

def get_cluster_label(doc_text, parent_label=None):
    prompt = f"Abstract:\n{doc_text}\n"
    if parent_label:
        prompt += f"Parent cluster label: {parent_label}\n"
    prompt += "Generate a concise and accurate cluster label with no extra text."
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        label = json.loads(response.text)["response"].strip()
        print("API call successful, generated label:", label)
    except Exception as e:
        label = f"API Error: {str(e)}"
    return label

if __name__ == "__main__":
    sample_text_path = "processed_texts\\acdc-a-structured-efficient-linear-layer_ICLR_2016.txt"
    with open(sample_text_path, "r", encoding="utf-8") as f:
        sample_text = f.read()
    label = get_cluster_label(sample_text, parent_label="Computer Vision")
    print("Generated cluster label:", label)

