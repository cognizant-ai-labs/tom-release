# Copyright (c) 2021 Cognizant Digital Business, Cognizant AI Labs
# Issued under this Academic Public License: github.com/cognizant-ai-labs/tom-release/LICENSE.

"""
Class for skipinit residual blocks using FiLM
"""

import torch
import torch.nn as nn

from film_layer import VEFilmLayer

class FilmResBlock(nn.Module):

    def __init__(self,
                 context_size,
                 hidden_size,
                 dropout=0.0):
        super(FilmResBlock, self).__init__()

        self.film_layer = VEFilmLayer(hidden_size,
                                      hidden_size,
                                      context_size)

        self.dropout_layer = nn.Dropout(dropout)

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, z):

        identity = x

        x = torch.relu(x)
        x = self.film_layer(x, z)
        x = self.dropout_layer(x)
        x = self.alpha * x

        return identity + x
