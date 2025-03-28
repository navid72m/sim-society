# 🧠 Simulating Society: AI, Justice, and Feedback

**What does it mean to govern fairly?** What if society could debate policies, express dissatisfaction, and evolve its own values — not through elections, but through learning?

This project explores these questions by simulating a society using AI agents and adaptive policy feedback loops. It combines philosophical curiosity with machine learning tools to create a living system that negotiates fairness, meritocracy, and inclusion.

---

## 🌍 Project Vision

We live in a world of difficult trade-offs:
- Should we prioritize **productivity** or **equity**?
- Is **meritocracy** fair when starting conditions differ?
- Can we design systems that respond to social feedback?

This repository explores how **AI agents** representing social groups — with different levels of privilege, education, and economic capital — can be used to evaluate and shape policy.

The goal isn’t to find the perfect policy, but to:
- Understand the **tensions between competing values**
- Simulate **social dynamics and collective decision-making**
- Experiment with **feedback-driven governance**

---

## 🧪 Repository Structure

This repo contains multiple experiments under a shared vision:

```
sim-society/
├── experiments/
│   ├── hybrid_llm_feedback/       # Static agents scoring policies via LLM
│   │   ├── feedback_loop.py
│   │   ├── output/
│   │   └── README.md
│   ├── marl_voting/              # Agents learn and vote on policy proposals
│   │   ├── marl_voting.py
│   │   ├── results/
│   │   └── README.md
├── agents/                       # Shared agent logic and abstractions
├── README.md                     # (this file)
└── requirements.txt
```

Each folder under `experiments/` is a unique simulation approach, from prompt-based policy rating to agent-based learning.

---

## 🤖 Philosophical Foundations

This project is inspired by the intersection of:
- **Political philosophy** (Rawls, utilitarianism, pluralism)
- **Multi-agent systems** (AI agents with different interests)
- **AI alignment** (learning goals that reflect human preferences)

By treating social policies as dynamic, learnable entities, we turn governance into an **adaptive process** — one that can optimize over time, account for feedback, and adjust to different values.

The simulation is not a final model of justice — it is a **mirror** that shows us what gets lost when we choose one value over another.

---

## 📊 Experiments Overview

### 1. `hybrid_llm_feedback/`
> LLM agents (via Ollama) score and justify policies based on their group’s socioeconomic profile. The system updates policies via gradient ascent to improve overall satisfaction.

### 2. `marl_voting/`
> Each agent learns its own internal values. They vote on proposed policies. The winning policy is evaluated and agents update their preferences based on reward. A blend of democracy and reinforcement learning.

---

## 🔍 Why This Matters

- It tests how **conflicting values** interact over time
- It builds systems that can **listen to minority feedback**
- It lets us observe **moral and political failure modes** before real-world deployment

And perhaps most importantly:
> It reminds us that justice is not fixed — it's a process of continuous debate, learning, and reflection.

---

## 🧠 Author

Built by **Navid Mirnouri**  
[GitHub](https://github.com/navid72m) ・ [LinkedIn](https://www.linkedin.com/in/navid-mirnouri)

If you're passionate about simulation, ethics, or alignment research, feel free to reach out or contribute.

---

## 📜 License

MIT License — use it, build on it, debate it.

---

## 🧭 Tags

`#AI` `#Simulation` `#Justice` `#Governance` `#PoliticalPhilosophy` `#LLM` `#MultiAgentSystems` `#EthicsInAI` `#ReinforcementLearning`

