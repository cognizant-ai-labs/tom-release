# Copyright (c) 2021 Cognizant Digital Business, Cognizant AI Labs
# Issued under this Academic Public License: github.com/cognizant-ai-labs/tom-release/LICENSE.

"""
Class for FiLM layer to support modulation by variable embeddings (VEs),
implemented using Conv1D for efficient parallel processing.
"""

import torch.nn as nn


class VEFilmLayer(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 context_size):
        super(VEFilmLayer, self).__init__()

        self.scale_layer = nn.Conv1d(context_size,
                                     output_size,
                                     1)
        self.shift_layer = nn.Conv1d(context_size,
                                     output_size,
                                     1)
        self.value_layer = nn.Conv1d(input_size,
                                     output_size,
                                     1)

    def forward(self, x, z):

        return self.scale_layer(z) * \
               self.value_layer(x) + \
               self.shift_layer(z)
