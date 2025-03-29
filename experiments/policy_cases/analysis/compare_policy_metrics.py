# analysis/compare_policy_texts.py
"""
Compare two policy proposals using:
- Token Entropy
- Semantic Diversity (1 - cosine similarity)
Input: Two JSON files with "response" or "revised_statement"
"""

import json
import numpy as np
from collections import Counter
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load JSON responses
with open("../output/ubi_deliberation_log.json") as f:
    data_a = json.load(f)["revised_statement"]

with open("../output/single_llm_policy.json") as f:
    data_b = json.load(f)["response"]

POLICIES = {
    "Multi-Agent": data_a,
    "Single-LLM": data_b
}

def compute_metrics(text):
    tokens = text.lower().split()
    freqs = Counter(tokens)
    probs = np.array(list(freqs.values())) / sum(freqs.values())
    token_entropy = entropy(probs, base=2)

    sentences = [s for s in text.split(".") if s.strip()]
    embeddings = model.encode(sentences)
    pairwise_sim = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = 1 - cosine(embeddings[i], embeddings[j])
            pairwise_sim.append(sim)

    avg_sim = np.mean(pairwise_sim) if pairwise_sim else 0.0
    semantic_diversity = 1 - avg_sim

    return token_entropy, semantic_diversity

# Compare
for name, text in POLICIES.items():
    token_H, semantic_D = compute_metrics(text)
    print(f"\n📊 {name} Policy:")
    print(f"  Token Entropy: {token_H:.3f}")
    print(f"  Semantic Diversity: {semantic_D:.3f}")
