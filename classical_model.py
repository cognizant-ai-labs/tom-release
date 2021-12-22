# Copyright (c) 2021 Cognizant Digital Business, Cognizant AI Labs
# Issued under this Academic Public License: github.com/cognizant-ai-labs/tom-release/LICENSE.

"""
Class for classical deep multi-task learning models implemented
in the TOM framework.

This model is referred to as Deep Residual Multi-task Learning (DR-MTL)
in the paper, as it is the classical form of MTL implemented as a
deep residual network (https://arxiv.org/pdf/2010.02354.pdf).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from classic_res_block import ClassicResBlock



class ClassicalModel(nn.Module):

    def __init__(self,
                 context_size,
                 hidden_size,
                 num_core_layers,
                 dropout=0.0):
        super(ClassicalModel, self).__init__()

        # Create Core
        self.core_blocks = nn.ModuleList([])
        for i in range(num_core_layers):
            core_block = ClassicResBlock(hidden_size,
                                      dropout)
            self.core_blocks.append(core_block)

        # Create dropout layer
        self.dropout = nn.Dropout(dropout)


    def forward(self, input_batch, input_contexts, output_contexts):

        # Setup encoder inputs
        x = input_batch

        # Apply input VEs classically
        x = F.linear(input_batch, input_contexts[0])
        x = self.dropout(x)

        # Apply model core
        for block in self.core_blocks:
           x = block(x)

        # Apply output VEs classically
        x = self.dropout(x)
        x = F.linear(x, output_contexts[0].t())

        return x



if __name__ == '__main__':
    model = ClassicalModel(2, 128, 30)
    print(model)
