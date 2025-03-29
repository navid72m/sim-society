# experiments/policy_cases/run_simulated_society.py
"""
Simulates how agents respond to two different policy proposals (multi-agent vs. single-agent).
Tracks agent satisfaction and justifications over multiple time steps.
"""

import json
import os
import matplotlib.pyplot as plt
from mediator_pipeline import AIAgent, call_ollama

# Load both policies
with open("output/ubi_deliberation_log.json") as f:
    policy_multi = json.load(f)["revised_statement"]

with open("output/single_llm_policy.json") as f:
    policy_single = json.load(f)["response"]

POLICIES = {
    "Multi-Agent": policy_multi,
    "Single-LLM": policy_single
}

# Define simulated society
AGENTS = [
    AIAgent("low_income",       [0.1, 0.6, 0.1, 0.1, 0.1]),
    AIAgent("tech_worker",      [0.5, 0.1, 0.3, 0.05, 0.05]),
    AIAgent("retired",          [0.2, 0.2, 0.1, 0.4, 0.1]),
    AIAgent("entrepreneur",     [0.6, 0.05, 0.3, 0.01, 0.04]),
    AIAgent("displaced_worker", [0.15, 0.25, 0.05, 0.15, 0.4])
]

STEPS = 5
os.makedirs("output", exist_ok=True)
results = {}

# Prompt builder
def build_prompt(agent, policy_text):
    names = ["Meritocracy", "Fairness", "Efficiency", "Age Inclusion", "Loss Recovery"]
    prefs = "\n".join([f"- {name}: {val}" for name, val in zip(names, agent.policy_vector)])
    return (
        f"You are an agent representing the '{agent.name}' group.\n"
        f"Your values are:\n{prefs}\n\n"
        f"Here is the current policy proposal:\n{policy_text}\n\n"
        f"Rate your satisfaction with this policy on a scale from 0 to 1, and briefly explain your reasoning."
    )

# Run simulation for each policy
for policy_name, policy_text in POLICIES.items():
    results[policy_name] = []
    print(f"\n▶ Running simulation for {policy_name} Policy")

    for step in range(STEPS):
        step_data = {}
        print(f"\n  Step {step+1}:")
        for agent in AGENTS:
            prompt = build_prompt(agent, policy_text)
            response = call_ollama(prompt)
            print(f"    [{agent.name}] → {response}")

            # Extract score
            try:
                number = next(float(w) for w in response.split() if w.replace('.', '', 1).isdigit())
                score = round(min(max(number, 0.0), 1.0), 2)
            except:
                score = 0.0

            step_data[agent.name] = {
                "score": score,
                "justification": response
            }
        results[policy_name].append(step_data)

# Save
with open("output/simulation_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Plot
for policy_name, series in results.items():
    plt.figure(figsize=(8, 5))
    for agent in AGENTS:
        scores = [round(step[agent.name]["score"], 2) for step in series]
        plt.plot(range(1, STEPS+1), scores, label=agent.name)

    plt.title(f"Agent Satisfaction Over Time ({policy_name})")
    plt.xlabel("Simulation Step")
    plt.ylabel("Satisfaction (0–1)")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"output/satisfaction_{policy_name.replace(' ', '_').lower()}.png")
    plt.close()

print("\n✅ Simulation complete. Results with justifications saved and visualized.")
