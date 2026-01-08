from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

hf_hub_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", filename="config.json")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text1 = "I like machine learning"
text2 = "I enjoy learning machines"

inputs = tokenizer([text1, text2], padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state.mean(dim=1)

similarity = F.cosine_similarity(embeddings[0], embeddings[1], dim=0)

print(similarity.item())
