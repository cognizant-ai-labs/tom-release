# Copyright (c) 2021 Cognizant Digital Business, Cognizant AI Labs
# Issued under this Academic Public License: github.com/cognizant-ai-labs/tom-release/LICENSE.

"""
Class for TOM implemented as in Section 4 of https://arxiv.org/pdf/2010.02354.pdf.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core_res_block import CoreResBlock
from film_layer import VEFilmLayer
from film_res_block import FilmResBlock
from inv_film_layer import InvVEFilmLayer



class TravelingObserverModel(nn.Module):

    def __init__(self,
                 context_size,
                 hidden_size,
                 num_encoder_layers,
                 num_core_layers,
                 num_decoder_layers,
                 dropout=0.0):
        super(TravelingObserverModel, self).__init__()


        # Create Encoder
        self.encoder_film_layer = VEFilmLayer(1,
                                              hidden_size,
                                              context_size)
        self.encoder_blocks = nn.ModuleList([])
        for i in range(num_encoder_layers - 1):
            encoder_block = FilmResBlock(context_size,
                                         hidden_size,
                                         dropout)
            self.encoder_blocks.append(encoder_block)

        # Create Core
        self.core_blocks = nn.ModuleList([])
        for i in range(num_core_layers):
            core_block = CoreResBlock(hidden_size,
                                      dropout)
            self.core_blocks.append(core_block)

        # Create Decoder
        self.decoder_blocks = nn.ModuleList([])
        for i in range(num_decoder_layers - 1):
            decoder_block = FilmResBlock(context_size,
                                         hidden_size,
                                         dropout)
            self.decoder_blocks.append(decoder_block)

        self.decoder_film_layer = InvVEFilmLayer(hidden_size,
                                                 1,
                                                 context_size)

        # Create dropout layer
        self.dropout = nn.Dropout(dropout)


    def forward(self, input_batch, input_contexts, output_contexts):

        # Setup encoder inputs
        batch_size = input_batch.shape[0]
        x = input_batch.unsqueeze(1)
        z = input_contexts.expand(batch_size, -1, -1)

        # Apply encoder
        x = self.encoder_film_layer(x, z)
        x = self.dropout(x)
        for block in self.encoder_blocks:
            x = block(x, z)

        # Aggregate state over variables
        x = torch.sum(x, dim=-1, keepdim=True)

        # Apply model core
        for block in self.core_blocks:
           x = block(x)

        # Setup decoder inputs
        x = x.expand(-1, -1, output_contexts.shape[-1])
        z = output_contexts.expand(batch_size, -1, -1)

        # Apply decoder
        for block in self.decoder_blocks:
            x = block(x, z)
        x = self.dropout(x)
        x = self.decoder_film_layer(x, z)

        # Remove unnecessary channels dimension
        x = torch.squeeze(x, dim=1)

        return x



if __name__ == '__main__':
    model = TravelingObserverModel(2, 128, 3, 3, 3, 0.2)
    print(model)
