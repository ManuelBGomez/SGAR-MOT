"""Mask Instance Resolver (MIR) class. 
It allows to mount the transformer architecture and process it.
"""

import numpy as np
import torch
from torch import nn

from .positional_encoding import SpatioTemporalPositionalEncoding
from .transformer_layers import TransformerEncoder, TransformerEncoderLayer


class MIR(nn.Module):
    def __init__(self, args):
        # Parent class initialization:
        super(MIR, self).__init__()
        self.args = args

        ## Args:
        self.dim_embedding = args.dim_embedding
        self.dim_model = args.trans_dim
        self.activation = self._get_activation_fn(self.args.activation)

        self.build_mir()

    def _get_activation_fn(self, activation='relu'):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == 'silu':
            return nn.SiLU()
        raise RuntimeError("activation should be relu/gelu/tanh/silu, not {}".format(activation))

    def build_mir(self):
        """Function that creates MIR with the necessary layers."""
        self.pos_encoder = SpatioTemporalPositionalEncoding(self.dim_model, dropout=self.args.dropout_p,
                                                            batch_first=True, device="cuda")
        # Separation token:
        self.sep_token = torch.nn.Parameter(torch.randn(self.dim_model), requires_grad=True).cuda()
        # Decision token:
        self.dec_token = torch.nn.Parameter(torch.randn(self.dim_model), requires_grad=True).cuda()
        # Encoder:
        encoder_layers = TransformerEncoderLayer(d_model=self.dim_model, nhead=self.args.nhead, dim_feedforward=self.args.ff_size,
                                                 dropout=self.args.dropout_p, activation=self.activation, batch_first=True).cuda()
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=self.args.num_layer).cuda()
        # Projection + decoder:
        self.linproj = nn.Linear(self.dim_embedding, self.dim_model).cuda()
        self.decoder = nn.Sequential(nn.LayerNorm(self.dim_model),
                                     nn.Linear(self.dim_model, 1)).cuda()
        self.sigmoid = nn.Sigmoid().cuda()

    def assemble_input(self, mask_embeddings):
        batch_size = mask_embeddings.shape[0]
        # We will input only masks from three timesteps: detection, previous and current.
        # A decision token will be in charge of deciding the final output (if we accept or not the mask).
        input_positions = {"MSK": None, "DEC": None}

        # Format [MS1] [MS2] [MS3] [DEC]

        # Append dec_token to the mask embeddings:
        dec_token = self.dec_token.repeat(batch_size, 1).unsqueeze(1)  # Shape B x 1 x 1
        mask_embeddings = torch.cat([mask_embeddings, dec_token], dim=1)

        num_masks = mask_embeddings.shape[1]

        # Input mask embeddings, without separators:
        input_mask_embeddings = torch.cat([torch.cat([mask_embeddings[:, [i], :]], dim=1) for i in range(num_masks)], dim=1)
        candidate_start = 0  # Without bbox embedding, we will directly begin in position 0.

        input_positions["MSK"] = tuple([candidate_start + i] for i in range(1, num_masks * 2 + 1, 2))
        input_positions["DEC"] = -1

        return input_mask_embeddings, input_positions

    def retrieve_output(self, output, input_positions):
        return output[:, input_positions["DEC"]]

    def forward(self, mask_embeddings, msk_bboxes, det_frame_id, t_frame_id):
        # Define the number of masks:
        batch_size, num_masks, ce_c, ce_h, ce_w = mask_embeddings.shape

        # Collapse channel, width and height dimensions:
        mask_embeddings_flat = mask_embeddings.view(batch_size, num_masks, -1)

        # Apply linear projection:
        mask_embeddings = self.linproj(mask_embeddings_flat) * np.sqrt(self.dim_model)

        # Assemble input:
        input_msk_seq, input_positions = self.assemble_input(mask_embeddings=mask_embeddings)

        # Positional encoding:
        total_num_masks = num_masks + 1
        transformer_input = self.pos_encoder(msk=input_msk_seq, msk_bboxes=msk_bboxes, num_masks=total_num_masks,
                                             det_frame_id=det_frame_id, t_frame_id=t_frame_id)

        # Apply encoder:
        transformer_output = self.transformer_encoder(transformer_input)

        # Retrieve output in the positions of interest:
        output = self.retrieve_output(output=transformer_output, input_positions=input_positions)

        # Decoder:
        output = self.decoder(output)  # B x 1 shape.

        return self.sigmoid(output)

    def load_pretrained(self, path):
        if not torch.cuda.is_available():
            state_dict = torch.load(path, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(path)

        if 'model_state_dict' in state_dict.keys():
            model_state_dict = state_dict['model_state_dict']
        else:
            model_state_dict = state_dict

        model_dict = self.state_dict()
        model_dict.update(model_state_dict)
        self.load_state_dict(model_dict)

    @property
    def num_params(self):
        """Extracts the number of parameters from the transformer."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self):
        """Extracts the number of trainable parameters from the transformer."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
