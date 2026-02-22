import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from unitraj.models.bevtraj.linear import build_mlp, MLP, FFN
from unitraj.models.bevtraj.utility import gen_sineembed_for_position


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def inverse_tanh(x, eps=1e-5):
    x = x.clamp(min=-1 + eps, max=1 - eps)
    return torch.atanh(x)


class QueryConditionedDynamics(nn.Module):
    """
    dynamics: [batch_size, query_num, dyn_dim]
    query_emb: [batch_size, query_num, query_dim]
    FiLM: Feature-wise Linear Modulation
    
    """
    def __init__(self, query_dim, hidden_dim):
        super().__init__()
        
        self.modulator = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        
        # Initialize the last layer to produce zero modulation at the beginning
        nn.init.zeros_(self.modulator[-1].weight)
        nn.init.zeros_(self.modulator[-1].bias)
        
        
    def forward(self, dynamics, query_emb):
        B, N, _ = query_emb.shape
        gamma_beta = self.modulator(query_emb) #(B, N, 2 * hidden_dim)
        gamma, beta = gamma_beta.chunk (2, dim=-1) #(B, N, hidden_dim) each
        conditioned = gamma * dynamics + beta #(B, N, hidden_dim)
        return conditioned
    

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
        
        self.pos_scale = build_mlp(self.D, self.D, self.D, dropout=self.dropout)
        self.query_pos = build_mlp(self.D, self.D, self.D, dropout=self.dropout)
        
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
        
    def forward(self, ba_query, query_sine_embed, ref_pos, bev_feat, lid):
        
        # Self-Attention
        pos_scale = self.pos_scale(ba_query) if lid != 0 else 1
        self_query_pos = pos_scale * self.query_pos(query_sine_embed)
        
        tgt = self.with_pos_embed(ba_query, self_query_pos)
        tgt, _ = self.self_attn(tgt, tgt, ba_query)
        tgt = self.norm_layers[0](ba_query + self.dropout_layers[0](tgt))
        
        # Cross-Attention
        cross_query_pos = query_sine_embed
        
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, cross_query_pos), ref_pos, bev_feat)
        tgt2 = self.norm_layers[1](tgt + self.dropout_layers[1](tgt2))
        
        output = self.norm_layers[2](self.ffn(tgt2))
        return output
        

class BEVDeformableAggregation(nn.Module):
    def __init__(self, config, d_model):
        super(BEVDeformableAggregation, self).__init__()
        
        self.config = config
        self.t = config['past_len']
        self.D = d_model
        self.t_D = config['t_dims']
        self.dropout = config['dropout']
        self.num_ba_query = config['num_ba_query']
        self.grid_size = config['grid_size']
        self.refine_share_param = config['refine_share_param']
        
        self.ba_query = nn.Parameter(torch.zeros(self.num_ba_query, self.D), requires_grad=True)

        self.dynamics_enc = nn.ModuleDict({
            'ec': MLP(self.config['target_attr'], self.D, self.t_D, 2),
            'ec_t': MLP(self.t_D * self.t, self.D // 2, self.D, 2),
            'ec_q': QueryConditionedDynamics(self.D, self.D),
            'ec_to_pos': MLP(self.D, self.D, 2, 2),
            'tc': MLP(self.config['target_attr'], self.D, self.t_D, 2),
            'tc_t': MLP(self.t_D * self.t, self.D // 2, self.D, 2),
            'tc_q': QueryConditionedDynamics(self.D, self.D),
            'hyb': MLP(self.D + self.D, self.D, 2, 3),
        })
        
        self.bda_layers = nn.ModuleList([
            BDALayer(self.config['bda_layer'], self.D, self.grid_size)
            for _ in range(self.config['num_bda_layers'])
        ])
        
        refine_layer = MLP(self.D, self.D, 2, 3)
        nn.init.zeros_(refine_layer.layers[-1].weight)
        nn.init.zeros_(refine_layer.layers[-1].bias)
        
        if self.refine_share_param:
            self.ref_pos_refine = refine_layer
        else:
            self.ref_pos_refine = _get_clones(refine_layer, self.config['num_bda_layers'])
            
        self.register_buffer('denorm_scale', torch.tensor(self.grid_size, dtype=torch.float32))
        
    def forward(self, bev_feat, ec_dyn, tc_dyn):
        B = bev_feat.shape[0]
        output = self.ba_query.repeat(B, 1, 1)
        
        ec_d = self.dynamics_enc['ec'](ec_dyn).reshape(B, -1) # (B, t_D * t)
        ec_d = self.dynamics_enc['ec_t'](ec_d).unsqueeze(0).expand(self.num_ba_query, -1, -1) # (num_ba_query, B, D)
        ec_d = self.dynamics_enc['ec_q'](ec_d.permute(1, 0, 2), output) # (B, num_ba_query, D)
        
        ec_pos = self.dynamics_enc['ec_to_pos'](ec_d).tanh()
        ec_pos= gen_sineembed_for_position(ec_pos, hidden_dim=self.D, temperature=10000)
        
        tc_d = self.dynamics_enc['tc'](tc_dyn).reshape(B, -1) # (B, t_D * t)
        tc_d = self.dynamics_enc['tc_t'](tc_d).unsqueeze(0).expand(self.num_ba_query, -1, -1) # (num_ba_query, B, D)
        tc_d = self.dynamics_enc['tc_q'](tc_d.permute(1, 0, 2), output) # (B, num_ba_query, D)
                    
        ec_ref_pos = self.dynamics_enc['hyb'](torch.cat([tc_d, ec_pos], dim=-1)).tanh() # (B, num_ba_query, 2)
        
        for lid, layer in enumerate(self.bda_layers):
            query_sine_embed = gen_sineembed_for_position(ec_ref_pos, hidden_dim=self.D, temperature=10000)
            
            output = layer(output, query_sine_embed, ec_ref_pos, bev_feat, lid)
            
            if self.refine_share_param:
                tmp = self.ref_pos_refine(output)
            else:
                tmp = self.ref_pos_refine[lid](output)
                
            ec_ref_pos = torch.tanh(tmp * 0.5 + ec_ref_pos)
            
        ref_pos_out = ec_ref_pos * self.denorm_scale[None, None, :]
        return output, ref_pos_out