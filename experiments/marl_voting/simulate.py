# experiments/marl_voting/simulate.py

from agent_groups import get_agent_groups
from marl_voting import generate_policy_candidates, tally_votes
import torch
import pandas as pd
import os

NUM_ROUNDS = 20
CANDIDATES_PER_ROUND = 5
POLICY_DIM = 5

# Create results directory if needed
os.makedirs("results", exist_ok=True)

agents = get_agent_groups()
log = []

for step in range(NUM_ROUNDS):
    proposals = generate_policy_candidates(CANDIDATES_PER_ROUND)
    votes = [agent.vote(proposals) for agent in agents]
    winning_index = tally_votes(votes)
    winning_policy = proposals[winning_index]

    for i, agent in enumerate(agents):
        reward = agent.evaluate(winning_policy)
        agent.learn_from_reward(reward, winning_policy)

        log_entry = {
            "step": step,
            "agent": agent.name,
            "reward": reward,
            "vote_index": votes[i],
            "winning_index": winning_index
        }
        # Log normalized policy vector values
        normalized_vector = torch.softmax(agent.policy_vector.detach(), dim=0).numpy()
        for j in range(POLICY_DIM):
            log_entry[f"policy_{j}"] = normalized_vector[j]

        log.append(log_entry)

# Save log
pd.DataFrame(log).to_csv("results/scores.csv", index=False)
print("✅ Simulation complete. Results saved to results/scores.csv")
