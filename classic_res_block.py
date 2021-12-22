# Copyright (c) 2021 Cognizant Digital Business, Cognizant AI Labs
# Issued under this Academic Public License: github.com/cognizant-ai-labs/tom-release/LICENSE.

"""
Class for skipinit residual blocks that do not use FiLM
"""

import torch
import torch.nn as nn


class ClassicResBlock(nn.Module):

    def __init__(self,
                 hidden_size,
                 dropout=0.0):
        super(ClassicResBlock, self).__init__()

        self.dense_layer = nn.Linear(hidden_size,
                                     hidden_size)

        self.dropout_layer = nn.Dropout(dropout)

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        identity = x

        x = torch.relu(x)
        x = self.dense_layer(x)
        x = self.dropout_layer(x)
        x = self.alpha * x

        return identity + x
