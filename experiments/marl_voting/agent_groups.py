# experiments/marl_voting/agent_groups.py

from marl_voting import LearningAgent

# Predefined group stats from UCI Adult-like categories
# These are simplified averages for prototyping

def get_agent_groups():
    groups = [
        LearningAgent("worker_female", {
            "education": 10.2,
            "income": 18000,
            "hours": 38,
            "age": 37,
            "loss": 0
        }),
        LearningAgent("worker_male", {
            "education": 10.5,
            "income": 25000,
            "hours": 42,
            "age": 39,
            "loss": 0
        }),
        LearningAgent("high_education", {
            "education": 16.1,
            "income": 70000,
            "hours": 43,
            "age": 41,
            "loss": 0
        }),
        LearningAgent("low_income", {
            "education": 9.0,
            "income": 15000,
            "hours": 35,
            "age": 33,
            "loss": 2000
        }),
        LearningAgent("high_income", {
            "education": 14.5,
            "income": 90000,
            "hours": 45,
            "age": 45,
            "loss": 0
        })
    ]
    return groups
