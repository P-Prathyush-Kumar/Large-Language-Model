import torch
import torch.nn.functional as F

# Sentence
sentence = ["this", "is", "a", "cat"]

# Dummy embeddings for each word (dimension 4 for simplicity)
embeddings = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],  # this
    [0.0, 2.0, 0.0, 2.0],  # is
    [1.0, 1.0, 0.0, 0.0],  # a
    [0.0, 0.0, 2.0, 1.0],  # cat
])

# Number of words
N, d = embeddings.shape

# Initialize weight matrices for Q, K, V (for single-head attention)
torch.manual_seed(0)
W_Q = torch.rand(d, d)
W_K = torch.rand(d, d)
W_V = torch.rand(d, d)

# Compute Q, K, V
Q = embeddings @ W_Q
K = embeddings @ W_K
V = embeddings @ W_V

# Compute attention scores (QK^T / sqrt(d))
scores = Q @ K.T / (d ** 0.5)

# Softmax to get attention weights
weights = F.softmax(scores, dim=-1)

# Compute output = weights * V
output = weights @ V

# Print results
print("Sentence:", sentence)
print("\nAttention Scores (QK^T / sqrt(d)):\n", scores.detach().numpy())
print("\nAttention Weights (after softmax):\n", weights.detach().numpy())
print("\nAttention Output for each word:\n", output.detach().numpy())
