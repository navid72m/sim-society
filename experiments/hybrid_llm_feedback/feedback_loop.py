# policy_loop/feedback_loop.py

"""
Simulates a feedback-driven policy optimization loop:
1. Load UCI Adult dataset and generate agent profiles
2. Initialize a complex policy vector
3. Feed it to agents and collect approval scores via LLM (Ollama)
4. Form coalitions among agents based on similar interests
5. Aggregate feedback by coalition
6. Optimize the policy using gradient ascent
7. Track and visualize coalition dynamics over time
8. Save agent responses to CSV
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.base_agent import Agent
import matplotlib.pyplot as plt
import csv
import os

# Load UCI Adult Dataset and prepare group stats
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income"
]
data = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)
data.dropna(inplace=True)

# Encode categorical variables
data["sex"] = pd.Categorical(data["sex"]).codes  # Male:1, Female:0

# Create agent groups based on real data
agents = []
group_definitions = {
    "worker_female": data[data["sex"] == 0],
    "worker_male": data[data["sex"] == 1],
    "high_education": data[data["education_num"] > 12],
    "low_income": data[data["income"] == " <=50K"],
    "high_income": data[data["income"] == " >50K"]
}

for name, group in group_definitions.items():
    avg_stats = {
        "education": group["education_num"].mean(),
        "income": (group["capital_gain"] + 1).mean(),
        "hours": group["hours_per_week"].mean(),
        "age": group["age"].mean(),
        "loss": group["capital_loss"].mean()
    }
    agents.append(Agent(name.capitalize(), name, avg_stats))

# Define a more complex policy vector with 5 dimensions:
# [meritocracy, fairness, efficiency, age inclusion, loss recovery]
initial_policy = [0.2, 0.2, 0.2, 0.2, 0.2]
policy_vector = torch.tensor(initial_policy, dtype=torch.float32, requires_grad=True)
optimizer = optim.Adam([policy_vector], lr=0.05)

# Track coalition scores over time
score_history = defaultdict(list)

# Prepare CSV file for logging responses
csv_path = "agent_responses.csv"
with open(csv_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Step", "Agent", "Role", "Score", "Explanation"])

    # Optimization loop
    for step in range(10):  # Fewer steps due to LLM latency
        optimizer.zero_grad()

        # Normalize policy vector using softmax
        normalized_policy = torch.softmax(policy_vector, dim=0)
        np_policy = normalized_policy.detach().numpy()

        # Collect scores and justifications from agents
        coalition_feedback = defaultdict(list)
        print(f"\nStep {step+1}: Policy = {np_policy}")
        for agent in agents:
            score, explanation = agent.llm_response(np_policy)
            print(f"{agent.name} ({agent.role}): {score:.2f} — {explanation}")
            coalition_feedback[agent.role].append(score)

            # Write to CSV
            # Write to CSV with cleaned explanation (single line, no newlines)
            cleaned_explanation = " ".join(explanation.strip().splitlines()).replace("\t", " ")
            writer.writerow([step + 1, agent.name, agent.role, f"{score:.2f}", cleaned_explanation])


        # Coalition average scores
        coalition_scores = {role: np.mean(scores) for role, scores in coalition_feedback.items()}
        for role, avg in coalition_scores.items():
            print(f"Coalition '{role}' average score: {avg:.2f}")
            score_history[role].append(avg)

        # Reward: weighted average of coalition scores
        avg_score = np.mean(list(coalition_scores.values()))
        loss = -torch.tensor(avg_score, requires_grad=True)
        loss.backward()
        optimizer.step()

        print(f"Overall Average Coalition Score: {avg_score:.2f}")

# Plot coalition score trends
plt.figure(figsize=(10, 6))
for role, scores in score_history.items():
    plt.plot(scores, label=role)
plt.xlabel("Simulation Step")
plt.ylabel("Average Coalition Score")
plt.title("Coalition Dynamics Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"\nAgent responses saved to: {csv_path}")