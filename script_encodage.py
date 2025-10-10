from transformers import pipeline

model_name = "Qwen/Qwen3-Embedding-0.6B"

embedder = pipeline("feature-extraction", model=model_name, trust_remote_code=True, device=-1)
embeddings = []
with open("DATASET6/final.csv","r") as f:
    for l in f:
        embeddings.append(embedder(l))

#transform list of tensors into a single tensor
import torch
embeddings = torch.tensor(embeddings).squeeze(1)
        
