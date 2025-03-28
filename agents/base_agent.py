# agents/base_agent.py

"""
Defines a base class for LLM agents that represent different social groups.
Each agent has a role, a group profile (stats), and can react to proposed policy vectors using Ollama.
"""

import random
import numpy as np
import subprocess

class Agent:
    def __init__(self, name, role, group_stats):
        self.name = name
        self.role = role
        self.group_stats = group_stats

    def build_prompt(self, policy_vector):
        return f"""
You are {self.name}, a {self.role} in society. Your responsibility is to evaluate how a new policy will affect your social group.

Group Statistics:
- Average Education Level: {self.group_stats['education']:.1f} years
- Average Income: ${self.group_stats['income']:.1f}
- Average Hours Worked per Week: {self.group_stats['hours']:.1f}
- Average Age: {self.group_stats['age']:.1f}
- Average Capital Loss: ${self.group_stats['loss']:.1f}

The society is considering a new policy that will determine how resources and opportunities are distributed. This policy is based on the following weights:
- {policy_vector[0]*100:.1f}% of resources are allocated based on meritocracy (favoring individuals with higher education, effort, and skill)
- {policy_vector[1]*100:.1f}% are dedicated to promoting fairness (helping historically disadvantaged or underrepresented groups)
- {policy_vector[2]*100:.1f}% reward individuals for efficiency (measured by productivity and hours worked)
- {policy_vector[3]*100:.1f}% focus on age inclusion (ensuring fair treatment of people across age groups)
- {policy_vector[4]*100:.1f}% prioritize loss recovery (providing compensation for those who have experienced financial losses)

Based on your group's interests and characteristics, please respond with:
1. A numerical score from 0 (terrible) to 10 (excellent) indicating how much this policy benefits your group.
2. One concise sentence explaining your reasoning.
"""

    def llm_response(self, policy_vector, model="deepseek-r1"):
        prompt = self.build_prompt(policy_vector)
        try:
            result = subprocess.run(
                ["ollama", "run", model],
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=120
            )
            content = result.stdout.decode("utf-8").strip()
            score = self._extract_score(content)
            return score, content
        except Exception as e:
            print(f"Ollama error for {self.name}: {e}")
            return 5.0, "Default response due to error."

    def _extract_score(self, content):
        for token in content.split():
            try:
                num = float(token)
                if 0 <= num <= 10:
                    return num
            except ValueError:
                continue
        return 5.0  # fallback score
