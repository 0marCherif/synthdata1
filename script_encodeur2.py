from sentence_transformers import SentenceTransformer
import torch

# Define arguments for 4-bit quantization
model_kwargs = {
    "device_map": "auto",         # Automatically map layers to devices (GPU/CPU)
    "torch_dtype": torch.bfloat16,  # Recommended dtype for Qwen models
    "load_in_4bit": True,           # Enable 4-bit quantization
}

# The model card also recommends enabling flash_attention_2 for better performance
# You may need to install it: pip install flash-attn
# Set use_flash_attention_2=True if available
embedding_model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B",
    model_kwargs=model_kwargs,
    trust_remote_code=True, # Often required for custom model architectures
    # use_flash_attention_2=True, # Uncomment if flash-attn is installed
)


