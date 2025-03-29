# experiments/policy_cases/experiment_ubi.py
"""
Deliberation experiment on Universal Basic Income (UBI).
Each agent represents a stakeholder group with distinct values.
The AI Mediator facilitates consensus-building through LLM-based synthesis and critique.
"""

from mediator_pipeline import AIAgent, AIMediator, call_ollama
import json
import os
import matplotlib.pyplot as plt

os.makedirs("output", exist_ok=True)
LOG_FILE = "experiments/policy_cases/output/ubi_deliberation_log.json"
PLOT_FILE = "experiments/policy_cases/output/ubi_satisfaction_plot.png"

POLICY_QUESTION = (
    "The policy question under discussion is:\n"
    "\"Should the government implement a universal basic income (UBI) "
    "to address economic inequality and automation-driven job loss?\""
)

# Extended agent class with topic prompt injection
def inject_topic(prompt: str, agent_name: str) -> str:
    return f"{POLICY_QUESTION}\n\nAgent: {agent_name}\n\n{prompt}"

class AIAgentWithTopic(AIAgent):
    def generate_opinion(self) -> str:
        opinion = call_ollama(inject_topic(super().generate_opinion(), self.name))
        print(f"\n[{self.name}] Opinion:\n{opinion}\n")
        return opinion

    def critique_statement(self, statement: str) -> str:
        critique = call_ollama(inject_topic(super().critique_statement(statement), self.name))
        print(f"\n[{self.name}] Critique:\n{critique}\n")
        return critique

    def evaluate_statement(self, statement: str) -> float:
        raw_prompt = inject_topic(statement, self.name)
        response = call_ollama(
            f"{raw_prompt}\n\nRate your satisfaction on a scale from 0 (very dissatisfied) to 1 (very satisfied). Only return a number."
        )
        print(f"\n[{self.name}] Satisfaction Response:\n{response}\n")
        try:
            return round(float(response), 2)
        except:
            return 0.0

# Define agents with diverse group value vectors
agents = [
    AIAgentWithTopic("low_income",       [0.1, 0.6, 0.1, 0.1, 0.1]),
    AIAgentWithTopic("tech_worker",      [0.5, 0.1, 0.3, 0.05, 0.05]),
    AIAgentWithTopic("retired",          [0.2, 0.2, 0.1, 0.4, 0.1]),
    AIAgentWithTopic("entrepreneur",     [0.6, 0.05, 0.3, 0.01, 0.04]),
    AIAgentWithTopic("displaced_worker", [0.15, 0.25, 0.05, 0.15, 0.4])
]

mediator = AIMediator()
log = {}

# Step 1: Opinions
opinions = [agent.generate_opinion() for agent in agents]
log["opinions"] = dict(zip([a.name for a in agents], opinions))

# Step 2: Initial Statement
initial_statement = mediator.synthesize_group_statement(opinions)
print("\n[Mediator] Initial Statement:\n", initial_statement, "\n")
log["initial_statement"] = initial_statement

# Step 3: Critiques
critiques = [agent.critique_statement(initial_statement) for agent in agents]
log["critiques"] = dict(zip([a.name for a in agents], critiques))

# Step 4: Revised Statement
revised_statement = mediator.revise_statement(initial_statement, critiques)
print("\n[Mediator] Revised Statement:\n", revised_statement, "\n")
log["revised_statement"] = revised_statement

# Step 5: Evaluation
scores = {agent.name: agent.evaluate_statement(revised_statement) for agent in agents}
log["final_scores"] = scores

# Output
with open(LOG_FILE, "w") as f:
    json.dump(log, f, indent=2)

# Plot
plt.figure(figsize=(8, 5))
names = list(scores.keys())
values = list(scores.values())
plt.bar(names, values, color="lightgreen")
plt.ylim(0, 1)
plt.ylabel("Satisfaction Score")
plt.title("Agent Satisfaction on UBI Policy")
plt.tight_layout()
plt.savefig(PLOT_FILE)
plt.close()

print(f"\n✅ UBI experiment complete. Results saved to {LOG_FILE} and {PLOT_FILE}")
