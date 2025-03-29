# 🧠 Simulating Society with Deliberative AI

This repository contains a research prototype exploring **deliberative AI** for **policy evaluation** using simulated societies of language model agents. We aim to model fairness, disagreement, and value alignment through agent-based reasoning and dynamic policy adaptation.

---

## 📜 Overview
Inspired by political philosophy (Habermas, Rawls) and modern interpretability methods, we build a system where agents with diverse values:

- Deliberate over social policy
- Provide critiques and justifications
- Influence policy evolution via feedback

The system compares **multi-agent policy generation** with **single-LLM generation**, measuring alignment, fairness, and reasoning quality.

---

## 📁 Key Experiments

### ✅ Experiment 1: UBI Policy Formation
- Agents (e.g., low-income, tech worker, retired) evaluate UBI
- Multi-agent deliberation produces policy via critique + revision
- Single-LLM baseline generated via direct prompt

### 🔄 Experiment 2: Policy Simulation Over Time
- Simulated society evaluates each policy across 5 steps
- Multi-agent policy evolves via agent feedback
- Single-LLM policy remains static

Metrics:
- Satisfaction scores
- Fairness (variance)
- Stability over time

### 📊 Experiment 3: Diversity Metrics
- **Token Entropy** for lexical richness
- **Embedding Diversity** for semantic variety

Multi-agent systems produced higher diversity and fairer satisfaction distributions.

### 🔬 Experiment 4: Mechanistic Interpretability Design

We design upcoming experiments to inspect *how* and *why* group deliberation works:

- Trace activation pathways via TransformerLens
- Identify "value circuits" (fairness, meritocracy)
- Measure error detection in solo vs. multi-agent critique

Planned Tasks:
- **Value steering**: amplify alignment dimensions
- **Error correction**: insert flaws, trace response
- **Consensus tracing**: identify convergence patterns

---

## 💡 Why It Matters

- Aligns LLMs with multi-dimensional social values
- Enables simulation of dissent, negotiation, and compromise
- Builds bridges between interpretability and democratic AI

This is a sandbox for AI alignment, governance research, and interactive simulations of social reasoning.

---

## 📌 Structure
```
sim-society/
├── experiments/
│   ├── hybrid_llm_feedback/
│   ├── marl_voting/
│   ├── policy_cases/
├── habermas_machine/      # Agent + mediator architecture
├── analysis/              # Metrics, plots, logs
├── README.md
```

---

## 🧠 Built With
- Python + PyTorch
- DeepSeek-R1 / LLaMA (via Ollama)
- UCI Adult dataset
- TransformerLens (planned)
- Matplotlib, CSV logs

---

## ✨ Credits
Designed and developed by **Navid Mirnouri**

GitHub: [navid72m](https://github.com/navid72m)  
LinkedIn: [Navid Mirnouri](https://www.linkedin.com/in/navid-mirnouri)

---

## 🔖 Tags
`#DeliberativeAI` `#AIAlignment` `#MultiAgentSystems` `#PolicySimulation` `#MechanisticInterpretability` `#Fairness` `#Ethics`

