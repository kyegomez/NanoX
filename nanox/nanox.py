import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ModuleNotFoundError:
    from torch.nn import LayerNorm
import copy

import torch.distributed as dist
from torch.hub import load_state_dict_from_url



#utils
def init_nanox_params(module):
    #init weights
    
    def normal_(data):
        #fsdp => module params will be on cuda => back to cpu
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))
    
    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zeros_()
    
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.proj.weight.data)
        normal_(module.v_proj.weight.data)


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


##############


class MultiwayNetwork(nn.Module):
    def __init__(self, module, dim=1):
        super().__init__()
        self.dim = dim
        self.A = module
        self.B = copy.deepcopy(module)
        self.B.reset_parameters()
        self.split_position = -1

    def forward(self, x, **kwargs):
        if self.split_position == -1:
            return self.A(x, **kwargs)
        if self.split_position == 0:
            return self.B(x, **kwargs)
        x1, x2 = torch.split(
            x,
            [self.split_position, x.size(self.dim) - self.split_position],
            dim=self.dim,
        )
        # x1, x2 = x[:self.split_position], x[self.split_position:]
        y1, y2 = self.A(x1, **kwargs), self.B(x2, **kwargs)
        return torch.cat([y1, y2], dim=self.dim)


class MutliwayEmbedding(MultiwayNetwork):
    def __init__(self, modules, dim=1):
        super(MultiwayNetwork, self).__init__()
        self.dim = dim
        assert len(modules) == 2
        self.A = modules[0]
        self.B = modules[1]
        self.split_position = -1




class XPOS(nn.Module):
    def __init__(
        self, head_dim, scale_base=512
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]
        
        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        num_heads,
        dropout=0.0,
        self_attention=False,
        encoder_decoder_attention=False,
        subln=False,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert self.self_attention ^ self.encoder_decoder_attention

        self.k_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.v_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.q_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.out_proj = MultiwayWrapper(
            args, nn.Linear(embed_dim, embed_dim, bias=True)
        )
        self.inner_attn_ln = (
            MultiwayWrapper(args, LayerNorm(self.embed_dim, eps=args.layernorm_eps))
            if subln and self.self_attention
            else None
        )
        self.dropout_module = torch.nn.Dropout(dropout)
        self.xpos = (
            XPOS(self.head_dim, args.xpos_scale_base)
            if args.xpos_rel_pos and self.self_attention
            else None
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key,
        value,
        incremental_state=None,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
        is_first_step=False,
    ):
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz * self.num_heads, src_len, self.head_dim)
        v = v.reshape(bsz * self.num_heads, src_len, self.head_dim)

        if incremental_state is not None:
            if "prev_key" in incremental_state:
                prev_key = incremental_state["prev_key"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                prev_value = incremental_state["prev_value"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                k = torch.cat([prev_key, k], dim=1)
                v = torch.cat([prev_value, v], dim=1)
            incremental_state["prev_key"] = k.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            incremental_state["prev_value"] = v.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            src_len = k.size(1)

        if self.xpos is not None:
            if incremental_state is not None and not is_first_step:
                offset = src_len - 1
            else:
                offset = 0
            k = self.xpos(k, offset=0, downscale=True)
            q = self.xpos(q, offset=offset, downscale=False)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights)
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if rel_pos is not None:
            rel_pos = rel_pos.view(attn_weights.size())
            attn_weights = attn_weights + rel_pos

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).reshape(tgt_len, bsz, embed_dim).transpose(0, 1)

        if self.inner_attn_ln is not None:
            attn = self.inner_attn_ln(attn)

        attn = self.out_proj(attn)
        attn_weights = attn_weights.view(
            bsz, self.num_heads, tgt_len, src_len
        ).transpose(1, 0)

        return attn, attn_weights
    

def MultiwayWrapper(args, module, dim=1):
    if args.multiway:
        return MultiwayNetwork(module, dim=dim)
    return module


def set_split_position(position):
    def apply_fn(module):
        if hasattr(module, "split_position"):
            module.split_position = position

    return apply_fn


def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x)
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\

def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m

def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


PRETRAINED_MODEL_URLS = {
  "pcqm4mv1_graphormer_base":"https://ml2md.blob.core.windows.net/graphormer-ckpts/checkpoint_best_pcqm4mv1.pt",
  "pcqm4mv2_graphormer_base":"https://ml2md.blob.core.windows.net/graphormer-ckpts/checkpoint_best_pcqm4mv2.pt",
  "oc20is2re_graphormer3d_base":"https://szheng.blob.core.windows.net/graphormer/modelzoo/oc20is2re/checkpoint_last_oc20_is2re.pt", # this pretrained model is temporarily unavailable
  "pcqm4mv1_graphormer_base_for_molhiv":"https://ml2md.blob.core.windows.net/graphormer-ckpts/checkpoint_base_preln_pcqm4mv1_for_hiv.pt",
}


def load_pretrained_model(model_name):
    if model_name not in PRETRAINED_MODEL_URLS:
        raise ValueError(f"IN load_pretrained_model => UNKOWN model name {model_name}")
    if not dist.is_initialized():
        return load_state_dict_from_url(PRETRAINED_MODEL_URLS[model_name], progress=True)["model"]
    else:
        pretrained_model = load_state_dict_from_url(PRETRAINED_MODEL_URLS[model_name], progress=True, file_name=f"{model_name}_{dist.get_rank()}")["model"]
        dist.barrier()
        return pretrained_model

#### model


class NanoNodeFeature(nn.Module):
    #compute note for each node in graph
    def __init__(
            self,
            num_heads, 
            num_atoms,
            num_in_degree,
            num_out_degree,
            hidden_dim,
            n_layers
    ):
        super(NanoNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.num_atoms = num_atoms

        #graph token
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
            num_out_degree, hidden_dim, padding_idx=0
        )

        self.graph_token = nn.Embedding(1, hidden_dim)
        self.apply(lambda module: init_params(module, n_layers=n_layers))
    

    def forward(self, batched_data):
        x, in_degree, out_degree = (
            batched_data["x"],
            batched_data["in_degree"],
            batched_data["out_degree"],
        )
        n_graph, n_node = x.size()[:2]
        
        #node feature + graph token
        node_feature = self.atom_encoder(x).sum(dim=-2) # [n_graph, n_node, n_hidden]

        node_feature = (
            node_feature
            + self.in_degree_encoder(in_degree),
            + self.out_degree_encoder(out_degree)
        )

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        return graph_node_feature
    


class NanoBias(nn.Module):
    #compute attn bias for each head
    def __init__(
            self,
            num_heads,
            num_atoms,
            num_edges,
            num_spatial,
            num_edge_distancetance,
            hidden_dim,
            edge_type,
            multi_hop_max_dist,
            n_layers,
    ):
        super(NanoBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist

        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        self.edge_type = edge_type
        if self.edge_type == "multihop":
            self.edge_dis_encoder = nn.Embedding(
                num_edge_distancetance * num_heads * num_heads, 1
            )
        self.spatial_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        bias, spatial_position, x = (
            batched_data["bias"],
            batched_data["spatial_position"],
            batched_data["x"],
        )
        edge_input, attn_edge_type = (
            batched_data["edge_input"],
            batched_data["attn_edge_type"],
        )

        n_graph, n_node = x.size()[:2]
        graph_bias = bias.clone()
        graph_bias = graph_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        ) # [n_graph, n_head, n_node+1, n_node+1]

        #spatial position
        #[n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_position_bias = self.spatial_position_encoder(spatial_position).permute(0, 3, 1, 2)
        graph_bias[:, :, 1: 1:] = graph_bias[:, :, 1:, 1:] + spatial_position_bias

        #reset spatial position here
        reshaped = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_bias[:, :, 1:, 0] = graph_bias[:, :, 1:, 0] + reshaped
        graph_bias[:, :, 0, :] = graph_bias[:, :, 0, :] + reshaped

        #edge feature
        if self.edge_type == "multi_hop":
            spatial_position_ = spatial_position.clone()
            spatial_position_[spatial_position_ == 0] = 1 #set pad to 1
            spatial_position_ = torch.where(spatial_position_ > 1, spatial_position_ - 1, spatial_position_)
            if self.multi_hop_max_dist > 0:
                spatial_position_ = spatial_position_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
            #[n_graph, n_node, n_node, max_dist, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.num_heads
            )
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(
                -1, self.num_heads, self.num_heads
                )[:max_dist, :, :],
            )
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heas
            ).permute(1, 2, 3, 0, 4)
            edge_input = (
                edge_input.sum(-2) / (spatial_position_.float().unsqueeze(-1))
            ).permute(0, 3, 1, 2)
        else:
            #[n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node, ]
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        graph_bias[:, :, 1:, 1:] = graph_bias[:, :, 1:, 1:] + edge_input
        graph_bias = graph_bias + bias.unsqueeze(1) # reset
        return graph_bias
    







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




#############

class NanoXGraphEncoder(nn.Module):
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
        


######################



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
        self.graph_encoder = NanoXGraphEncoder()
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