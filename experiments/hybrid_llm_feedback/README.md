# 🧠 Hybrid Society Simulation with LLM Agents

This project simulates a society composed of AI agents representing different social groups. These agents evaluate public policies based on real-world data and provide feedback. The system then adjusts its policies through a feedback-driven optimization loop to increase satisfaction across coalitions.

---

## 📌 Features

- Uses real demographic and economic data (UCI Adult Dataset)
- Defines agents by group-level statistics (e.g., income, education, age)
- Represents social policy as a 5-dimensional vector:
  - **Meritocracy**: Rewarding education and skill
  - **Fairness**: Supporting disadvantaged groups
  - **Efficiency**: Encouraging productivity
  - **Age Inclusion**: Considering fairness across age groups
  - **Loss Recovery**: Compensating prior financial losses
- Each agent uses an LLM (via [Ollama](https://ollama.com)) to:
  - Score the policy (0–10)
  - Explain its reasoning from its group's perspective
- The system optimizes policies using gradient ascent
- Logs agent responses and plots coalition satisfaction trends

---

## 🛠️ Tech Stack

- Python 3.8+
- PyTorch (for optimization)
- Ollama (local LLM inference)
- Matplotlib (visualization)
- Pandas, NumPy

---

## 📂 Structure

```
.
├── agents/
│   └── base_agent.py         # Agent definition and prompt logic
├── policy_loop/
│   └── feedback_loop.py      # Main simulation loop and visualization
├── output/
│   ├── agent_responses.csv   # Logged responses per step
│   └── coalition_plot.png    # Coalition satisfaction plot
├── requirements.txt
└── README.md
```

---

## 🚀 Run the Simulation

### 1. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Pull an LLM (e.g., Llama3):
```bash
ollama pull llama3
```

### 3. Run the simulation:
```bash
python policy_loop/feedback_loop.py
```

This runs a 10-step policy evolution process. LLM agents provide feedback at each step, and results are logged and visualized.

---

## 📊 Output

- **agent_responses.csv**: Each agent’s score, role, and justification per round
- **coalition_plot.png**: Shows average satisfaction of each group over time

---

## 📈 Results and Analysis

### Coalition Dynamics Over Time

![Coalition Plot](Coalition%20.png)

This plot visualizes how each coalition (social group) responded to evolving policies:

- **worker_female (blue)**: Peaks quickly at max satisfaction (10), drops briefly at step 7, then recovers. This suggests policies heavily favored fairness or age inclusion.
- **worker_male (orange)**: Persistently low satisfaction. Policies likely ignored their group needs or scored poorly on efficiency/meritocracy.
- **high_education (green)**: Flat and low, indicating underrepresentation in most policy allocations.
- **low_income (red)**: Highly reactive. Spikes in satisfaction when fairness or loss recovery is prioritized; crashes when meritocracy rises.
- **high_income (purple)**: Brief peak at step 3, followed by disappointment. Suggests one-off benefit from a temporary efficiency-focused policy.

### Key Insight
This plot shows the inherent trade-offs in policy making: some groups benefit at the cost of others. A fair and stable system must account for these dynamics.

---

## 👍 Example Prompt

Each agent receives a prompt like:

> "You are a low-income worker. Your average group income is $18k. The policy allocates 30% to fairness and 20% to meritocracy..."
>
> *Score: 7*
>
> *Explanation: This policy helps us by addressing systemic disadvantages and compensating prior economic harm.*

---

## 🌍 Why This Matters

This project explores how AI can simulate democratic processes and trade-offs. It models:

- Ethical conflict between values
- Feedback-driven governance
- Multi-agent social negotiation using LLMs

---

## 📌 Next Steps

- Add voting or rebellion mechanisms
- Train agents to adapt via reinforcement learning (MARL)
- Introduce human policymaker in the loop

---

## 🧠 Author

Developed by **Navid Mirnouri**  
[GitHub](https://github.com/navid72m) · [LinkedIn](https://www.linkedin.com/in/navid-mirnouri)

---

## 📄 License

MIT License — open to research, extension, and remixing.

---

## 📌 Tags

`#AI` `#LLM` `#Governance` `#Simulation` `#PoliticalPhilosophy` `#Ollama` `#EthicsInAI` `#SocialAgents`

