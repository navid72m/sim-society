# agent_learning/marl_voting.py
"""
Hybrid MARL + Voting Simulation

Each agent:
- Has a learnable internal policy vector (preferences over 5 policy dimensions)
- Votes for a global policy based on similarity to proposals
- Evaluates global policy based on alignment with group interests
- Learns to adapt its internal policy using simple REINFORCE-like updates

This extends the previous Hybrid Society Simulation with agent learning and decentralized decision-making.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

POLICY_DIM = 5  # [meritocracy, fairness, efficiency, age inclusion, loss recovery]

class LearningAgent:
    def __init__(self, name, group_stats, lr=0.01):
        self.name = name
        self.group_stats = group_stats
        self.policy_vector = nn.Parameter(torch.rand(POLICY_DIM))
        self.optimizer = optim.Adam([self.policy_vector], lr=lr)

    def vote(self, policy_candidates):
        # Choose policy with highest cosine similarity to internal policy
        with torch.no_grad():
            similarities = [
                torch.nn.functional.cosine_similarity(self.policy_vector, cand, dim=0).item()
                for cand in policy_candidates
            ]
            return int(np.argmax(similarities))

    def evaluate(self, global_policy):
        """
        Heuristic satisfaction score based on alignment to agent's group stats.
        Can be replaced with LLM-based scoring.
        """
        weights = global_policy.detach().numpy()
        score = (
            weights[0] * self.group_stats['education'] +
            weights[1] * (1 if self.group_stats['income'] < 50000 else 0) +
            weights[2] * self.group_stats['hours'] +
            weights[3] * (1 - abs(self.group_stats['age'] - 40) / 40) +
            weights[4] * self.group_stats['loss']
        )
        return score / 100.0  # Normalize

    def learn_from_reward(self, reward, selected_policy):
        self.optimizer.zero_grad()
        similarity = torch.nn.functional.cosine_similarity(self.policy_vector, selected_policy, dim=0)
        loss = -reward * similarity  # Encourage alignment with rewarding policy
        loss.backward()
        self.optimizer.step()


def generate_policy_candidates(n=5):
    return [torch.softmax(torch.rand(POLICY_DIM), dim=0) for _ in range(n)]

def tally_votes(votes):
    return max(set(votes), key=votes.count)

# Next steps:
# - Create real agents using UCI stats
# - Run simulation loop
# - Log scores and policy drift
# - Compare learned policies across agent types
