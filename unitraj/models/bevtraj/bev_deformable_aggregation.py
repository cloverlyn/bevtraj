import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import pickle
from abc import ABC, abstractmethod

from unitraj.models.bevtraj.linear import build_mlp, MLP, FFN
from unitraj.models.bevtraj.positional_encoding_utils import gen_sineembed_for_position


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def inverse_tanh(x, eps=1e-5):
    x = x.clamp(min=-1 + eps, max=1 - eps)
    return torch.atanh(x)

class DeformAttn(nn.Module):
    def __init__(self, config, d_model, grid_size):
        super(DeformAttn, self).__init__()
        
        self.config = config
        self.D = d_model
        self.n_heads = config['num_heads']
        self.n_points = config['num_key_points']
        self.head_dims = self.D // self.n_heads
        
        self.sampling_offsets = build_mlp(self.D, self.D, self.n_heads * self.n_points * 2, dropout=0.0)
        self.attn_weights = build_mlp(self.D, self.D, self.n_heads * self.n_points, dropout=0.0)
        self.value_proj = nn.Conv2d(self.head_dims, self.head_dims, kernel_size=1)
        self.output_proj = build_mlp(self.D, self.D, self.D, dropout=0.0)
        
        self.register_buffer('offset_normalizer', torch.tensor(grid_size, dtype=torch.float32))
        
    def forward(self, ba_query, ref_pos, bev_feat):
        B, _, H, W = bev_feat.shape
        N = ba_query.shape[1]
        
        value = bev_feat.reshape(B, self.n_heads, -1, H, W).reshape(B*self.n_heads, -1, H, W)
        value = self.value_proj(value)
        
        sampling_offsets = self.sampling_offsets(ba_query).reshape(B, N, self.n_heads, self.n_points, 2).permute(0, 2, 1, 3, 4)
        sampling_locations = ref_pos.unsqueeze(1).unsqueeze(3) + sampling_offsets / self.offset_normalizer[None, None, None, None, :]
        sampling_locations = sampling_locations.reshape(B*self.n_heads, N, self.n_points, 2)
        
        sampling_locations[..., 1] = sampling_locations[..., 1] * -1 # Align with F.grid_sample coordinate system
        
        sampled_feature = F.grid_sample(value, sampling_locations, align_corners=False, mode='bilinear')
        sampled_feature = sampled_feature.reshape(B, self.n_heads, -1, N, self.n_points).permute(0, 1, 3, 4, 2)
        
        attn_weights = self.attn_weights(ba_query).reshape(B, N, self.n_heads, self.n_points)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.permute(0, 2, 1, 3).unsqueeze(-1)
        
        attn_outputs = torch.sum(sampled_feature * attn_weights, dim=3)
        attn_outputs = attn_outputs.permute(0, 2, 1, 3).reshape(B, N, -1)
        
        output = self.output_proj(attn_outputs)
        return output
        

class BDALayer(nn.Module):
    def __init__(self, config, d_model, grid_size):
        super(BDALayer, self).__init__()
        
        self.config = config
        self.D = d_model
        self.dropout = config['dropout']
        
        self.self_attn = nn.MultiheadAttention(self.D, config['num_heads'], dropout=self.dropout, batch_first=True)
        self.cross_attn = DeformAttn(config['deform_attn'], self.D, grid_size)
        self.ffn = FFN(self.D, config['ffn_dims'], num_fcs=2, dropout=self.dropout)
        
        self.norm_layers= nn.ModuleList([nn.LayerNorm(self.D) for _ in range(3)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(2)])
    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
        
    def forward(self, ba_query, query_pos, ref_pos, bev_feat):
        tgt = self.with_pos_embed(ba_query, query_pos)
        tgt, _ = self.self_attn(tgt, tgt, ba_query)
        tgt = self.norm_layers[0](ba_query + self.dropout_layers[0](tgt))
        
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), ref_pos, bev_feat)
        tgt2 = self.norm_layers[1](tgt + self.dropout_layers[1](tgt2))
        
        output = self.norm_layers[2](self.ffn(tgt2))
        return output
        

class BEVDeformableAggregation(nn.Module, ABC):
    def __init__(self, config, d_model):
        super(BEVDeformableAggregation, self).__init__()
        
        self.config = config
        self.D = d_model
        self.dropout = config['dropout']
        # self.num_ba_query = config['num_ba_query']
        self.num_ba_query = 512
        self.grid_size = config['grid_size']
        
        self.ba_query = nn.Parameter(torch.zeros(self.num_ba_query, self.D), requires_grad=True)
        
        self.pos_scale = build_mlp(self.D, self.D, self.D, dropout=self.dropout)
        self.query_pos = build_mlp(self.D, self.D, self.D, dropout=self.dropout)
        
        self.bda_layers = nn.ModuleList([
            BDALayer(self.config['bda_layer'], self.D, self.grid_size)
            for _ in range(self.config['num_bda_layers'])
        ])
            
        self.register_buffer('denorm_scale', torch.tensor(self.grid_size, dtype=torch.float32))
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    

class BDA_ENC(BEVDeformableAggregation):
    def __init__(self, config, d_model):
        super(BDA_ENC, self).__init__(config, d_model)
        
        self.refine_share_param = config['refine_share_param']
        
        file_path = 'unitraj/models/bevtraj_ms/cluster_64_center_dict_6s.pkl'
        with open(file_path, 'rb') as f:
            anchors = pickle.load(f)
        self.register_buffer('anchors', torch.from_numpy(anchors['VEHICLE']).float())
        
        refine_layer = MLP(self.D, self.D, 2, 3)
        nn.init.zeros_(refine_layer.layers[-1].weight)
        nn.init.zeros_(refine_layer.layers[-1].bias)
        
        if self.refine_share_param:
            self.ref_pos_refine = refine_layer
        else:
            self.ref_pos_refine = _get_clones(refine_layer, self.config['num_bda_layers'])
            
    def place_template_points(self, tc_pos, obj_pos, obj_heading):
        """
        Place template points around each object by applying its pose (position + heading).

        Args:
            tc_pos (torch.Tensor): (B, A, K, 2) (target-centric coordinates)
            obj_pos (torch.Tensor): (B, A, 2) (x, y)
            obj_heading (torch.Tensor): (B, A, 2) (sin, cos)

        Returns:
            new_pos (torch.Tensor): (B, A, K, 2) template positions for each object
        """
        B, A, K, _ = tc_pos.shape

        sin, cos = obj_heading[..., 0], obj_heading[..., 1]  # (B,A)

        rot_mat = torch.stack([
            torch.stack([cos, sin], dim=-1),
            torch.stack([-sin, cos], dim=-1)
        ], dim=-2)   # (B,A,2,2)

        rotated = torch.matmul(tc_pos, rot_mat)  # (B,A,K,2)
        new_pos = rotated + obj_pos[:, :, None, :]   # (B,A,K,2)

        return new_pos
    
    def target_to_ego(self, tc_pos, ego_pos, ego_heading):
        """
        Transform points from target-centric coordinates to ego-centric coordinates.

        Args:
            tc_pos (torch.Tensor): (B, N, 2) (target-centric coordinates)
            ego_pos (torch.Tensor): (B, 2) (x, y)
            ego_heading (torch.Tensor): (B, 2) (sin, cos)

        Returns:
            ego_pos (torch.Tensor): (B, N, 2) points in ego-centric coordinates
        """
        B, N, _ = tc_pos.shape

        sin, cos = ego_heading[:, 0], ego_heading[:, 1]  # (B,)

        rot_mat = torch.stack([
            torch.stack([cos, -sin], dim=-1),
            torch.stack([sin, cos], dim=-1)
        ], dim=-2)   # (B,2,2)

        pos = tc_pos - ego_pos[:, None, :]   # (B,N,2)
        ego_centric_pos = torch.matmul(pos, rot_mat)  # (B,N,2)

        return ego_centric_pos
        
    def forward(self, traj_data, bev_feat):
        # create reference points based on object positions and headings
        obj_pos = traj_data['obj_trajs'][:, :8, -1, 0:2] # x, y
        obj_heading = traj_data['obj_trajs'][:, :8, -1, -6:-4] # sin, cos
        
        B, A, _ = obj_pos.shape
        K = self.anchors.shape[0]
        
        anchor_pos = self.anchors[None, None, :, :]
        anchor_pos = anchor_pos.repeat(B, A, 1, 1)
        
        ref_pos_target = self.place_template_points(anchor_pos, obj_pos, obj_heading).reshape(B, A*K, 2) # (B, 256, 2)
        ref_pos_ego = self.target_to_ego(ref_pos_target, obj_pos[:, 1, :], obj_heading[:, 1, :]) # (B, 256, 2) # kong_fixme
        ref_pos = ref_pos_ego / self.denorm_scale[None, None, :] # normalize
        
        # BEV Deformable Aggregation
        output = self.ba_query[None].repeat(B, 1, 1)
        
        for lid, layer in enumerate(self.bda_layers):
            ref_pos_denorm = ref_pos * self.denorm_scale[None, None, :]
            query_sine_embed = gen_sineembed_for_position(ref_pos_denorm, hidden_dim=self.D, temperature=10000)
            query_pos = self.pos_scale(output) * self.query_pos(query_sine_embed)
            
            output = layer(output, query_pos, ref_pos, bev_feat)
            
            if self.refine_share_param:
                tmp = self.ref_pos_refine(output)
            else:
                tmp = self.ref_pos_refine[lid](output)
                
            ref_pos = tmp.tanh() * 0.5 + ref_pos
            
        ref_pos_out = ref_pos * self.denorm_scale[None, None, :]
        return output, ref_pos_out
    
    
class BDA_DEC(BEVDeformableAggregation):
    def __init__(self, config, d_model):
        super(BDA_DEC, self).__init__(config, d_model)
        
        self.ref_pos = nn.Parameter(self.make_center_grid(self.num_ba_query), requires_grad=True)
        
    def make_center_grid(self, n_points):
        N = int(math.sqrt(n_points))
        assert N * N == n_points, "n_points must be a perfect square"

        centers = (torch.arange(N, dtype=torch.float32) + 0.5) / N * 2 - 1
        grid_y, grid_x = torch.meshgrid(centers, centers, indexing='ij')

        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.reshape(-1, 2)

        return grid
    
    def forward(self, bev_feat):
        B = bev_feat.shape[0]
        output, ref_pos = map(lambda x: x[None].repeat(B, 1, 1), [self.ba_query, self.ref_pos])
        
        for layer in self.bda_layers:
            query_sine_embed = gen_sineembed_for_position(ref_pos, hidden_dim=self.D, temperature=20)
            query_pos = self.pos_scale(output) * self.query_pos(query_sine_embed)
            
            output = layer(output, query_pos, ref_pos, bev_feat)
        
        ref_pos_out = ref_pos * self.denorm_scale[None, None, :]
        return output, ref_pos_out