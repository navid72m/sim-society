# neural_model/transformer.py

"""
Defines the SocialPolicyPredictor model — a simple feedforward neural network that
predicts a 3-element policy vector: [meritocracy_weight, fairness_weight, efficiency_weight].
"""

import torch
import torch.nn as nn

class SocialPolicyPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SocialPolicyPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)  # [merit, fairness, efficiency]
        )

    def forward(self, x):
        return self.network(x)
