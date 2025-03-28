# 🧠 Hybrid Society Simulation with LLM Agents

This project simulates a society where different social groups — powered by LLM agents — evaluate and respond to public policies. The system learns to improve policies based on agent feedback over time.

## 🚀 What It Does

- Creates AI agents based on real UCI Adult dataset demographics
- Represents social values using a policy vector:
  - Meritocracy
  - Fairness
  - Efficiency
  - Age Inclusion
  - Loss Recovery
- Each agent scores and explains how the policy affects their group
- The system adapts using a feedback loop to improve collective satisfaction

## 🛠️ Tech Stack

- Python + PyTorch
- Ollama (LLM interface)
- Matplotlib
- CSV + Pandas
- Gradient ascent optimization

## 📊 Output

- `agent_responses.csv`: All scores and justifications
- `coalition_plot.png`: Visualizes group satisfaction over time

## 📁 Structure

- `agents/`: Agent class and prompt builder
- `policy_loop/`: Simulation and optimization
- `neural_model/`: (Optional) Metric computations
- `output/`: All generated results

## 📌 Example

> “Score: 8  
> Explanation: The policy’s emphasis on meritocracy aligns well with my high education level…”

## 🔧 Setup

1. Install requirements:

```bash
pip install -r requirements.txt
