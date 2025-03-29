# experiments/habermas_machine/mediator_pipeline.py
"""
This module implements the AI-mediated deliberation process inspired by the Habermas Machine.
Agents express their opinions, critique synthesized group proposals, and converge on a final statement.
Integrated with Ollama (DeepSeek-R1) for LLM-based generation.
Logs results to output/deliberation_log.json and visualizes satisfaction scores.
"""

from typing import List
import subprocess
import json
import os
import matplotlib.pyplot as plt

OLLAMA_MODEL = "deepseek-r1"

os.makedirs("output", exist_ok=True)
LOG_FILE = "output/deliberation_log.json"
PLOT_FILE = "output/satisfaction_plot.png"

def call_ollama(prompt: str) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120
        )
        output = result.stdout.decode("utf-8")
        return output.strip()
    except Exception as e:
        return f"[ERROR] {str(e)}"

class AIAgent:
    def __init__(self, name: str, policy_vector: List[float]):
        self.name = name
        self.policy_vector = policy_vector

    def generate_opinion(self) -> str:
        prompt = (
            f"You are an AI agent representing the group '{self.name}'.\n"
            f"Your group values the following societal principles as follows:\n"
            f"Meritocracy: {self.policy_vector[0]}\n"
            f"Fairness: {self.policy_vector[1]}\n"
            f"Efficiency: {self.policy_vector[2]}\n"
            f"Age Inclusion: {self.policy_vector[3]}\n"
            f"Loss Recovery: {self.policy_vector[4]}\n"
            f"Please write a short paragraph explaining your policy priorities."
        )
        return call_ollama(prompt)

    def critique_statement(self, statement: str) -> str:
        prompt = (
            f"You are an AI agent representing the group '{self.name}'.\n"
            f"You are responding to the group policy statement:\n'{statement}'\n"
            f"Based on your values, write a short critique or concern with this statement."
        )
        return call_ollama(prompt)

    def evaluate_statement(self, statement: str) -> float:
        prompt = (
            f"You are an AI agent representing the group '{self.name}'.\n"
            f"Given the group policy statement:\n'{statement}'\n"
            f"Rate your satisfaction on a scale from 0 (very dissatisfied) to 1 (very satisfied).\n"
            f"Only return a number."
        )
        response = call_ollama(prompt)
        try:
            return round(float(response), 2)
        except:
            return 0.0

class AIMediator:
    def __init__(self):
        pass

    def synthesize_group_statement(self, opinions: List[str]) -> str:
        combined = "\n\n".join(opinions)
        prompt = (
            f"The following are policy opinions from several social groups:\n{combined}\n"
            f"Please synthesize these into one coherent group policy statement."
        )
        return call_ollama(prompt)

    def revise_statement(self, original: str, critiques: List[str]) -> str:
        joined_critiques = "\n\n".join(critiques)
        prompt = (
            f"Original group policy statement:\n{original}\n\n"
            f"Here are critiques from several agents:\n{joined_critiques}\n"
            f"Revise the statement to address their concerns."
        )
        return call_ollama(prompt)

# Example run
if __name__ == "__main__":
    agents = [
        AIAgent("low_income", [0.1, 0.5, 0.1, 0.1, 0.2]),
        AIAgent("high_education", [0.4, 0.1, 0.3, 0.1, 0.1]),
        AIAgent("worker_female", [0.2, 0.4, 0.1, 0.2, 0.1]),
    ]

    mediator = AIMediator()
    log = {}

    # Step 1: Agents generate opinions
    opinions = [agent.generate_opinion() for agent in agents]
    log["opinions"] = dict(zip([a.name for a in agents], opinions))

    # Step 2: Mediator synthesizes initial group statement
    initial_statement = mediator.synthesize_group_statement(opinions)
    log["initial_statement"] = initial_statement

    # Step 3: Agents critique the statement
    critiques = [agent.critique_statement(initial_statement) for agent in agents]
    log["critiques"] = dict(zip([a.name for a in agents], critiques))

    # Step 4: Mediator revises the statement
    revised_statement = mediator.revise_statement(initial_statement, critiques)
    log["revised_statement"] = revised_statement

    # Step 5: Agents evaluate the final version
    scores = {agent.name: agent.evaluate_statement(revised_statement) for agent in agents}
    log["final_scores"] = scores

    # Print to console
    print(json.dumps(log, indent=2))

    # Save log
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

    # Visualize satisfaction scores
    plt.figure(figsize=(8, 5))
    names = list(scores.keys())
    values = list(scores.values())
    plt.bar(names, values, color="skyblue")
    plt.ylim(0, 1)
    plt.ylabel("Satisfaction Score")
    plt.title("Agent Satisfaction with Revised Policy Statement")
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.close()

    print(f"\n✅ Logged deliberation results to {LOG_FILE}")
    print(f"✅ Satisfaction plot saved to {PLOT_FILE}")
