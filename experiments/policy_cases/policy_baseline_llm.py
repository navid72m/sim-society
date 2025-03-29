# experiments/policy_cases/policy_baseline_llm.py
"""
Generates a single-agent LLM baseline policy on the UBI question.
This simulates what a strong standalone language model would produce without deliberation.
"""

import os
import json
from mediator_pipeline import call_ollama

os.makedirs("./output", exist_ok=True)
BASELINE_FILE = "./output/single_llm_policy.json"

PROMPT = (
    "You are a highly capable AI policy advisor."
    " Consider the following policy question and propose a detailed, fair, and effective answer."
    "\n\n"
    "Policy Question: Should the government implement a universal basic income (UBI)"
    " to address economic inequality and automation-driven job loss?"
    "\n\n"
    "Your answer should reflect multiple considerations (fairness, meritocracy, economic productivity,"
    " age inclusion, and recovery for displaced workers). Please write a coherent, concise policy proposal."
)

policy_response = call_ollama(PROMPT)

print("\n🧠 Single-LLM Baseline Policy Proposal:\n")
print(policy_response)

with open(BASELINE_FILE, "w") as f:
    json.dump({"prompt": PROMPT, "response": policy_response}, f, indent=2)

print(f"\n✅ Saved single-agent policy to {BASELINE_FILE}")
