import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from nanox.model.pretained import load_pretrained_model
from nanox.model.encoder import NanoXEncoder

class NanoXModel(nn.Module):
    def __init__(self, encoder, encoder_embed_dim=1024, pretrained_model_name="none", load_pretrained_model_output_layer=True):
        super().__init__()
        self.encoder = encoder
        self.encoder_embed_dim = encoder_embed_dim
        if pretrained_model_name != "none":
            self.load_state_dict(load_pretrained_model(pretrained_model_name))
            if not load_pretrained_model_output_layer:
                self.encoder.reset_output_layer_parameters()

    def max_nodes(self):
        return self.encoder.max_nodes

    def forward(self, batched_data, **kwargs):
        return self.encoder(batched_data, **kwargs)


class NanoX(nn.Module):
    def __init__(self, max_nodes=512, share_input_output_embed=False, remove_head=False, activation_fn=nn.GELU()):
        super().__init__()
        self.max_nodes = max_nodes
        self.graph_encoder = NanoXEncoder()
        self.share_input_output_embed = share_input_output_embed
        self.embed_out = None
        self.lm_output_learned_bias = None
        self.load_softmax = not remove_head
        self.masked_lm_pooler = nn.Linear(1024, 1024)
        self.lm_head_transform_weight = nn.Linear(1024, 1024)
        self.activation_fn = activation_fn
        self.layer_norm = LayerNorm(1024)
        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(1024, 1000, bias=False)

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        if self.embed_out is not None:
            self.embed_out.reset_parameters()

    def forward(self, batched_data, perturb=None, masked_tokens=None, **unused):
        inner_states, graph_representation = self.graph_encoder(batched_data, perturb=perturb)
        x = inner_states[-1].transpose(0, 1)

        if masked_tokens is not None:
            raise NotImplementedError

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        if self.share_input_output_embed and hasattr(self.graph_encoder.embed_tokens, "weight"):
            x = F.linear(x, self.graph_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        return x

    def max_nodes(self):
        return self.max_nodes

    def upgrade_state_dict_named(self, state_dict, name):
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if "embed_out.weight" in k or "lm_output_learned_bias" in k:
                    del state_dict[k]
        return state_dict