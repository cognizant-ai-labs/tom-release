# Copyright (c) 2021 Cognizant Digital Business, Cognizant AI Labs
# Issued under this Academic Public License: github.com/cognizant-ai-labs/tom-release/LICENSE.

"""
Class for Inverted VE FiLM layer (used in decoder),
implemented using Conv1D for efficient parallel processing.
"""

import torch.nn as nn


class InvVEFilmLayer(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 context_size):
        super(InvVEFilmLayer, self).__init__()

        self.scale_layer = nn.Conv1d(context_size,
                                     input_size,
                                     1)
        self.shift_layer = nn.Conv1d(context_size,
                                     input_size,
                                     1)
        self.value_layer = nn.Conv1d(input_size,
                                     output_size,
                                     1)

    def forward(self, x, z):

        return self.value_layer(
                   self.scale_layer(z) * x + \
                   self.shift_layer(z))
