# analysis/simulate_both_policies.py
"""
Simulates both policies (multi-agent and single-agent) across a society of agents.
Tracks satisfaction and justifications over multiple steps.
"""

import json
import os
import sys
# import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mediator_pipeline import AIAgent, call_ollama
import matplotlib.pyplot as plt

# Load policies
with open("../output/ubi_deliberation_log.json") as f:
    policy_multi = json.load(f)["revised_statement"]

with open("../output/single_llm_policy.json") as f:
    policy_single = json.load(f)["response"]

POLICIES = {
    "Multi-Agent": policy_multi,
    "Single-LLM": policy_single
}

# Simulated agents
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

# Build prompts
def build_prompt(agent, policy):
    value_labels = ["Meritocracy", "Fairness", "Efficiency", "Age Inclusion", "Loss Recovery"]
    values = "\n".join([f"- {name}: {weight}" for name, weight in zip(value_labels, agent.policy_vector)])
    return (
        f"You are an agent representing the '{agent.name}' group.\n"
        f"Your group values are:\n{values}\n\n"
        f"Here is the policy proposal:\n{policy}\n\n"
        f"Rate your satisfaction with this policy from 0 to 1 and explain your reasoning."
    )

# Run simulations
for label, policy in POLICIES.items():
    print(f"\n🧪 Running simulation for {label} policy")
    results[label] = []
    for step in range(STEPS):
        step_result = {}
        print(f"  Step {step+1}:")
        for agent in AGENTS:
            prompt = build_prompt(agent, policy)
            response = call_ollama(prompt)
            try:
                score = float(next(w for w in response.split() if w.replace('.', '', 1).isdigit()))
                score = round(min(max(score, 0.0), 1.0), 2)
            except:
                score = 0.0
            step_result[agent.name] = {"score": score, "justification": response.strip()}
            print(f"    {agent.name}: {score} — {response.strip()[:60]}...")
        results[label].append(step_result)

# Save results
with open("output/policy_simulation_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Plot results
for label, series in results.items():
    plt.figure(figsize=(8, 5))
    for agent in AGENTS:
        scores = [step[agent.name]["score"] for step in series]
        plt.plot(range(1, STEPS+1), scores, label=agent.name)
    plt.title(f"Satisfaction Over Time — {label}")
    plt.xlabel("Step")
    plt.ylabel("Satisfaction")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"output/satisfaction_{label.replace(' ', '_').lower()}.png")
    plt.close()

print("\n✅ Simulation complete. Results saved and plots generated.")
