import math 
import torch 
import torch.nn as nn

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)



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
    

