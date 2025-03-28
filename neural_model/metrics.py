# neural_model/metrics.py

"""
Defines the function `compute_metrics()` to compute extended social metrics for each individual:
- Meritocracy: based on education, capital gain, and hours worked
- Fairness: group-level disparities
- Efficiency: normalized productivity
- Age Inclusion: favors individuals near societal average age
- Loss Recovery: captures economic compensation needs

Returns a NumPy array with shape (n_samples, 5)
"""

import numpy as np

def compute_metrics(data):
    # Meritocracy score: weighted sum
    meritocracy = (
        0.5 * data["education_num"] / data["education_num"].max() +
        0.3 * data["hours_per_week"] / data["hours_per_week"].max() +
        0.2 * data["capital_gain"] / (data["capital_gain"].max() + 1e-6)
    )

    # Efficiency score
    efficiency = data["hours_per_week"] / data["hours_per_week"].max()

    # Fairness score: based on gender disparity
    overall_avg = data["hours_per_week"].mean()
    male_avg = data[data["sex"] == 1]["hours_per_week"].mean()
    female_avg = data[data["sex"] == 0]["hours_per_week"].mean()
    disparity = abs(male_avg - female_avg) / data["hours_per_week"].max()
    fairness = 1 - disparity
    fairness_scores = np.full(len(data), fairness)

    # Age inclusion: closer to mean age is better
    mean_age = data["age"].mean()
    age_inclusion = 1 - np.abs(data["age"] - mean_age) / mean_age

    # Loss recovery: higher capital loss = higher need
    loss_recovery = data["capital_loss"] / (data["capital_loss"].max() + 1e-6)

    # Stack metrics together: shape (n_samples, 5)
    metrics = np.stack([meritocracy, fairness_scores, efficiency, age_inclusion, loss_recovery], axis=1)
    return metrics
