# Copyright (c) 2021 Cognizant Digital Business, Cognizant AI Labs
# Issued under this Academic Public License: github.com/cognizant-ai-labs/tom-release/LICENSE.

"""
Class for soft order models that fit in the VE framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class SoftOrderModel(nn.Module):

    def __init__(self,
                 hidden_size,
                 num_core_layers,
                 num_tasks,
                 dropout=0.0):
        super(SoftOrderModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_core_layers = num_core_layers
        self.num_tasks = num_tasks

        # Create Core
        self.core_layers = nn.ModuleList([])
        for i in range(num_core_layers):
            core_layer = nn.Linear(hidden_size, hidden_size)
            self.core_layers.append(core_layer)

        # Create soft order scaling parameters (S in Figure 3 of the SLO paper)
        self.scalars = nn.Parameter(torch.zeros(num_tasks,
                                                num_core_layers,
                                                num_core_layers))

        # Create dropout layer
        self.dropout = nn.Dropout(dropout)


    def forward(self, input_batch, input_contexts, output_contexts, task_idx):

        # Setup encoder inputs
        x = input_batch

        # Apply input encoder classically
        x = F.linear(input_batch, input_contexts[0])

        # Apply model core
        batch_size = input_batch.shape[0]
        task_scalars = self.scalars[task_idx]
        for depth in range(self.num_core_layers):
            depth_scalars = task_scalars[depth]
            soft_depth_scalars = F.softmax(depth_scalars, dim=0)
            layer_outputs = []
            for layer in range(self.num_core_layers):
                layer_output = self.core_layers[layer](x)
                layer_output = F.relu(layer_output)
                layer_output = self.dropout(layer_output)
                layer_output = layer_output * soft_depth_scalars[layer]
                layer_outputs.append(layer_output)
            stacked_layer_outputs = torch.stack(layer_outputs)
            x = torch.sum(stacked_layer_outputs, dim=0)

        # Apply output VEs classically
        x = F.linear(x, output_contexts[0].t())

        return x



if __name__ == '__main__':
    model = SoftOrderModel(128, 4, 100)
    print(model)
    print(list(model.parameters()))
