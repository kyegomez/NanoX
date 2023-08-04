import torch
import torch.nn as nn
from nanox.model.layers import NanoNodeFeature, NanoBias
from nanox.model.encoder_layer import NanoXGraphEncoderLayer

class NanXEncoder(nn.Module):
    def __init__(self, 
                num_atoms, 
                num_in_degree, 
                num_out_degree, 
                num_edges, num_spatial, 
                num_edge_distance, 
                edge_type, 
                multi_hop_max_dist,
                num_encoder_layers=12, 
                embedding_dim=768, 
                ffn_embedding_dim=768, 
                num_attention_heads=32, 
                dropout=0.1, 
                activation_fn=nn.ReLU(), 
                embed_scale=None, 
                freeze_embeddings=False, 
                n_trans_layers_to_freeze=0, 
                export=False, 
                traceable=False, 
                q_noise=0.0, 
                qn_block_size=8):
        super().__init__()
        self.dropout_module = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim
        self.traceable = traceable

        self.graph_node_feature = NanoNodeFeature(num_heads=num_attention_heads, 
                                                  num_atoms=num_atoms, 
                                                  num_in_degree=num_in_degree, 
                                                  num_out_degree=num_out_degree, 
                                                  hidden_dim=embedding_dim, 
                                                  n_layers=num_encoder_layers)
        
        self.graph_attn_bias = NanoBias(num_heads=num_attention_heads, 
                                        num_atoms=num_atoms, 
                                        num_edges=num_edges, 
                                        num_spatial=num_spatial, 
                                        num_edge_distance=num_edge_distance, 
                                        edge_type=edge_type, 
                                        multi_hop_max_dist=multi_hop_max_dist, 
                                        hidden_dim=embedding_dim, 
                                        n_layers=num_encoder_layers)
        
        self.embed = embed_scale

        if q_noise > 0:
            self.quant_noise = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        else:
            self.quant_noise = None

        self.emb_layer_norm = nn.LayerNorm(self.embedding_dim, elementwise_affine=export)
        self.layers = nn.ModuleList([NanoXGraphEncoderLayer(embedding_dim=embedding_dim, 
                                                            ffn_embedding_dim=ffn_embedding_dim, 
                                                            num_heads=num_attention_heads, 
                                                            dropout=dropout, 
                                                            activation_fn=activation_fn, 
                                                            export=export) for _ in range(num_encoder_layers)])

        if freeze_embeddings:
            for layer in range(n_trans_layers_to_freeze):
                for param in self.layers[layer].parameters():
                    param.requires_grad=True
    
    def forward(self, batched_data,
                pertub=None,
                last_state_only=False,
                token_embedding=None, 
                mask=None):
        data_x = batched_data["x"]
        n_graph, n_node = data_x.size()[:2]
        
        padding_mask = (data_x[:, :, 0]).eq(0)
        padding_mask_cls = torch.zeros(n_graph, 1, device=padding_mask.device, 
                                       dtype=padding_mask.dtype)
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)

        if token_embedding is not None:
            x = token_embedding
        else:
            x = self.graph_node_feature(batched_data)

        if pertub is not None:
            x[:, 1: :] *= pertub
        
        attn_bias = self.graph_attn_bias(batched_data)

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.quant_noise is not None:
            x  = self.quant_noise(x)
        
        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)
        
        x = self.dropout_module(x)

        x = x.tranpose(0, 1)

        inner_states = [x] if not last_state_only else []
        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask, self_attn_mask=mask, 
                         self_attn_bias=attn_bias)
            
            if not last_state_only:
                inner_states.append(x)
        
        graph_representation = x[0, :, :]

        if self.traceable:
            return torch.stack(inner_states), graph_representation
        else:
            return inner_states, graph_representation