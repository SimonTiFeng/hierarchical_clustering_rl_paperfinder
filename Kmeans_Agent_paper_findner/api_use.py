import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3"

def get_cluster_label(doc_text, parent_label=None):
    prompt = f"Abstract:\n{doc_text}\n"
    if parent_label:
        prompt += f"Parent cluster label: {parent_label}\n"
    prompt += "Generate a concise and accurate cluster label."
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        label = result.get("completion", "No label")
    except Exception as e:
        label = f"API Error: {str(e)}"
    return label

if __name__ == "__main__":
    sample_text = "This paper discusses a novel approach for deep learning-based image segmentation."
    label = get_cluster_label(sample_text, parent_label="Computer Vision")
    print(label)
