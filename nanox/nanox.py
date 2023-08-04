import logging
import torch 
import torch.nn.functional as F

from torch.nn import LayerNorm

class Nanox(nn.Module):
    def __init__(self,
                 args,
                 encoder):
        super().__init__()
        self.encoder = encoder
        self.args = args
        self.encoder_embed_dim = args.encoder_embed_dim
        if args_pretrained_model_name != "none":
            self.load_state_dict(load_pretrained_model(args.))