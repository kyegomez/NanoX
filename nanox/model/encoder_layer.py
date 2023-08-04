import torch.nn as nn
from nanox.model.multihead_attention import MultiheadAttention

class NanoXGraphEncoderLayer(nn.Module):
    def __init__(self, embedding_dim=768, 
                 ffn_embedding_dim=3072, 
                 num_heads=8, 
                 dropout=0.1, 
                 activation_fn=nn.relu, pre_layernorm=False):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.pre_layernorm = pre_layernorm

        self.dropout_module = nn.Dropout(dropout)
        self.activation_fn = activation_fn
        self.self_attn = MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None):
        residual = x
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(x, x, x, attn_mask=self_attn_mask, key_padding_mask=self_attn_padding_mask)
        x = self.dropout_module(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.dropout_module(x)
        x = self.fc2(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.final_layer_norm(x)
        return x, attn