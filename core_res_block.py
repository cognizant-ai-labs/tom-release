# Copyright (c) 2021 Cognizant Digital Business, Cognizant AI Labs
# Issued under this Academic Public License: github.com/cognizant-ai-labs/tom-release/LICENSE.

"""
Class for skipinit residual blocks without FiLM, implemented using
Conv1D in order to maintain TOM implementation pattern.
"""

import torch
import torch.nn as nn


class CoreResBlock(nn.Module):

    def __init__(self,
                 hidden_size,
                 dropout=0.0):
        super(CoreResBlock, self).__init__()

        self.conv_layer = nn.Conv1d(hidden_size,
                                    hidden_size,
                                    1)

        self.dropout_layer = nn.Dropout(dropout)

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        identity = x

        x = torch.relu(x)
        x = self.conv_layer(x)
        x = self.dropout_layer(x)
        x = self.alpha * x

        return identity + x
