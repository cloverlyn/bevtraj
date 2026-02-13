import math
import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from unitraj.models.bevtraj.bev_deformable_aggregation_dec import BEVDeformableAggregation
from unitraj.models.bevtraj.decoder_deform_attn2d import DeformableCrossAttention2D_Q
from unitraj.models.bevtraj.linear import MLP, FFN, MotionRegHead, MotionClsHead
from unitraj.models.bevtraj.positional_encoding_utils import gen_sineembed_for_position

class QueryConditionedDynamics(nn.Module):
    """
    dynamics: [batch_size, query_num, dyn_dim]
    query_emb: [batch_size, query_num, query_dim]
    FiLM: Feature-wise Linear Modulation
    
    """
    def __init__(self, dyn_dim, query_dim, hidden_dim):
        super().__init__()
        assert dyn_dim == hidden_dim, "For simplicity, dyn_dim should be equal to hidden_dim"
        
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
    
# =========================================================================================
# Utility: faster coordinate transform
# =========================================================================================
def ego_to_target(center_pos, t_x, t_y, r_s, r_c):
    """
    Fast target-centric coordinate transform.
    center_pos: (B, N, 2)
    returns: (B, N, 2)
    """
    B, N, _ = center_pos.shape
    trans = torch.stack([t_x, t_y], dim=-1)  # (B, 1, 2)
    center = center_pos + trans

    R = torch.stack([
        torch.cat([r_c,  r_s], dim=-1),
        torch.cat([-r_s, r_c], dim=-1),
    ], dim=1)  # (B, 2, 2)

    return center @ R


# =========================================================================================
# Utility: Positional encoding cache
# =========================================================================================
class SineEmbedCache:
    """Caches sine embeddings to avoid repeated PE computing (speed optimization)."""

    def __init__(self):
        self.cache = {}

    def get(self, key, pos, dim, T):
        if key in self.cache:
            return self.cache[key]
        out = gen_sineembed_for_position(pos, hidden_dim=dim, temperature=T)
        self.cache[key] = out
        return out


# =========================================================================================
# Temporal Positional Encoding (unchanged)
# =========================================================================================
class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, future_len=12, temperature=500.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(future_len, d_model)
        position = torch.arange(0, future_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(temperature) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# =========================================================================================
# Decoder Layer (reshape/permute minimized)
# =========================================================================================
class BEVTrajDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.T = config['future_len']
        self.D = config['d_model']
        self.Q_D = config['query_dims']
        self.ffn_D = config['ffn_dims']
        
        self.K = config['num_modes']
        self.dropout = config['dropout']
        self.num_heads = config['num_heads']
        self.spa_pos_T = config['spa_pos_T']
        
        self.to_pos_Q = MLP(self.Q_D, self.Q_D, self.Q_D, 2)
        self.norm = nn.ModuleList([nn.LayerNorm(self.D) for _ in range(3)])
        self.temp_self_attn = nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(self.D, self.num_heads,
                                                                    dim_feedforward=self.ffn_D, dropout=self.dropout)
        self.bev_cross_attn = DeformableCrossAttention2D_Q(**config['deform_cross_attn_query'])
        self.ffn = FFN(self.D, self.ffn_D, 2)
    
    def forward(self, dec_embed, scene_context, bev_feat, query_scale, ref_points, ego_dynamics):
        '''
        Args:
            dec_embed: [T, B*K, D]
            scene_context: [t, B, D]
            bev_feat: [B, D, H, W]
            query_scale: [T, B*K, d]
            ref_points: [K, B, T, 2]
        '''
        B = bev_feat.size(0)
        scene_context = scene_context 
        
        # ============================== target-centric(tc) modeling ==============================
        
        dec_embed = self.norm[0](self.temp_self_attn(query=dec_embed, key=dec_embed, value=dec_embed)[0] + dec_embed)
        
        # get positional query
        query_sine_embed = gen_sineembed_for_position(ref_points, hidden_dim=self.Q_D, temperature=self.spa_pos_T)
        tc_pos_Q = self.to_pos_Q(query_sine_embed)
        
        dec_embed, query_scale = map(lambda t: t.reshape(self.T, B, self.K, -1).permute(2, 1, 0, 3), (dec_embed, query_scale))
        dec_embed = dec_embed + tc_pos_Q
        dec_embed = self.transformer_decoder_layer(tgt=dec_embed.reshape(self.K, B*self.T, -1),
                                                   memory=scene_context).reshape(self.K, B, self.T, -1)
        
        # ============================== ego-centric(ec) modeling ==============================
        
        # coord transform
        ego_loc, ego_sin, ego_cos = ego_dynamics['ego_loc'], ego_dynamics['ego_sin'], ego_dynamics['ego_cos']
        ego_loc, ego_sin, ego_cos = map(lambda t: t.unsqueeze(1).unsqueeze(0).repeat(self.K, 1, self.T, 1), (ego_loc, ego_sin, ego_cos))
        
        rotation_matrix = torch.stack([
            torch.cat([ego_cos, -ego_sin], dim=-1),
            torch.cat([ego_sin, ego_cos], dim=-1)
        ], dim=-2)
        
        ref_points = ref_points - ego_loc
        ref_points = torch.matmul(ref_points.unsqueeze(-2), rotation_matrix).squeeze(-2)
        
        # cross attn with bev feature
        dec_embed = self.norm[1](self.bev_cross_attn(dec_embed, bev_feat, query_scale, ref_points))
        dec_embed = self.norm[2](self.ffn(dec_embed))
        
        return dec_embed


# =========================================================================================
# Main Decoder (full optimization)
# =========================================================================================
class BEVTrajDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # short refs
        self.t = config['past_len']
        self.T = config['future_len']
        self.D = config['d_model']
        self.ffn_D = config['ffn_dims']
        self.t_D = config['t_dims']
        self.T_D = config['T_dims']
        self.K = config['num_modes']
        self.target_attr = config['target_attr']
        self.query_scale_dims = config['query_scale_dims']
        self.tem_pos_T = config['tem_pos_T']
        self.spa_pos_T = config['spa_pos_T']
        self.dropout = config['dropout']
        self.L_goal_proposal = config['num_goal_proposal_layers']
        self.L_dec = config['num_decoder_layers']
        self.num_heads = config['num_heads']
        
        self.dca_k_cfg = config['deform_cross_attn_key']
        self.dca_q_cfg = config['deform_cross_attn_query']
        self.dca_k_cfg['dim'] = self.dca_q_cfg['dim'] = self.D
        
        self.dec_layer_config = {
            'future_len': self.T,
            'd_model': self.D,
            'query_dims': self.query_scale_dims,
            'ffn_dims': self.ffn_D,
            'spa_pos_T': self.spa_pos_T,
            'num_modes': self.K,
            'dropout': self.dropout,
            'num_heads': self.num_heads,
            'deform_cross_attn_query': self.dca_q_cfg,
        }
        
        # ============================ Learnable mode queries ============================
        self.Q = nn.Parameter(torch.empty(self.K, 1, self.D))
        nn.init.xavier_uniform_(self.Q)

        # ============================ BDA modules ============================
        self.bda_sgcp = BEVDeformableAggregation(self.config['bda_sgcp'], self.D)

        # ============================ Target dynamics encoder ============================
        self.tc_dynamic_encoder = nn.ModuleDict({
            'enc':   MLP(self.config['target_attr'], self.D, self.t_D, 2),
            'enc_s': MLP(self.t_D * self.t, self.D // 2, self.D, 2),
            'enc_s_mode': MLP(self.D * 2, self.D, self.D, 2),
            'enc_c': MLP(self.t_D * self.t, self.D // 2, self.D, 2),
            'enc_c_mode': MLP(self.D * 2, self.D, self.D, 2),
        })

        # ============================ Goal proposal stack ============================
        self.goal_proposal = nn.ModuleList()
        for _ in range(config['num_goal_proposal_layers']):
            self.goal_proposal.append(
                nn.ModuleDict({
                    'self_attn': nn.MultiheadAttention(self.D, config['num_heads'],
                                                       dropout=config['dropout']),
                    'norm1': nn.LayerNorm(self.D),

                    'cross_attn': nn.MultiheadAttention(self.D * 2, config['num_heads'],
                                                        dropout=config['dropout'],
                                                        kdim=self.D * 2, vdim=self.D),
                    'q_proj': MLP(self.D * 2, self.D, self.D, 2),
                    'norm2': nn.LayerNorm(self.D),

                    'ffn': FFN(self.D, self.ffn_D, 2),
                    'norm3': nn.LayerNorm(self.D),

                    'goal_reg': MLP(self.D, self.D, 2, 2),
                })
            )

        self.goal_FDE = MLP(self.D, self.D, 1, 2)

        # ============================ Initial Prediction ============================
        self.get_query_scale_l1 = MLP(self.D, self.query_scale_dims, self.query_scale_dims, 2)
        self.norm_l1 = nn.ModuleList([nn.LayerNorm(self.D) for _ in range(3)])
        
        self.context_cross_attn_l1 = nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout)
        self.bev_cross_attn_l1 = DeformableCrossAttention2D_Q(**self.dca_q_cfg)
        self.ffn_l1 = FFN(self.D, self.ffn_D, 2)
        
        self.tmp_MLP = nn.ModuleList([
            nn.Sequential(nn.Linear(self.D, self.T_D * self.T), nn.GELU()),
            nn.Sequential(nn.Linear(self.T_D, self.D), nn.GELU())
        ])
        self.motion_cls_l1 = MotionClsHead(self.D, self.T_D, self.T)
        self.motion_reg_l1 = MotionRegHead(self.D)

        # ============================ Iterative Refinement ============================
        
        self.temp_pos_enc = TemporalPositionalEncoding(self.D, self.dropout, future_len=self.T, temperature=self.tem_pos_T)
        self.get_query_scale_T = MLP(self.query_scale_dims, self.query_scale_dims, self.query_scale_dims, 2)
        
        dec_layer = BEVTrajDecoderLayer(self.dec_layer_config)
        self.dec_layers = nn.ModuleList([copy.deepcopy(dec_layer) for _ in range(self.L_dec - 1)])
        
        self.motion_cls = MotionClsHead(self.D, self.T_D, self.T)
        self.motion_reg = MotionRegHead(self.D)
        

    # ================================================================================
    # Goal Candidate Proposal (optimized, no behavior change)
    # ================================================================================
    def goal_candidate_proposal(self, tc_dyn, bda_token, bda_pos):
        B = tc_dyn.size(0)

        # dynamic feature encoding (query positional encoding)
        d = self.tc_dynamic_encoder['enc'](tc_dyn).reshape(B, -1)
        d_s = self.tc_dynamic_encoder['enc_s'](d).unsqueeze(0).expand(self.K, -1, -1)
        d_s = self.tc_dynamic_encoder['enc_s_mode'](torch.cat([self.Q.expand(-1, B, -1), d_s], dim=-1))
        d_c = self.tc_dynamic_encoder['enc_c'](d).unsqueeze(0).expand(self.K, -1, -1)
        d_c = self.tc_dynamic_encoder['enc_c_mode'](torch.cat([self.Q.expand(-1, B, -1), d_c], dim=-1))

        # BEV pos encoding (cached)
        pos_k = SineEmbedCache().get("bda_pos", bda_pos, self.D, self.config['spa_pos_T'])
        k = torch.cat([bda_token, pos_k], dim=-1).permute(1, 0, 2)
        v = bda_token.permute(1, 0, 2)

        goal_reg = None

        for i, layer in enumerate(self.goal_proposal):
            if i==0:
                src = self.Q.expand(-1, B, -1)
                mode_q = self.Q.expand(-1, B, -1) + d_s
                
            else:
                src = mode_q
                goal_reg_pos = SineEmbedCache().get("goal_reg_pos", goal_reg, self.D, self.config['spa_pos_T'])
                mode_q = mode_q + goal_reg_pos
            
            mode_q = layer['self_attn'](mode_q, mode_q, mode_q)[0] + src
            mode_q = layer['norm1'](mode_q)
            
            if i==0:
                q = torch.cat([mode_q, d_c], dim=-1)
            
            else:
                q = torch.cat([mode_q, goal_reg_pos], dim=-1)


            q = layer['cross_attn'](q, k, v)[0]
            q = layer['q_proj'](q)
            mode_q = layer['norm2'](q + mode_q)
            mode_q = layer['norm3'](layer['ffn'](mode_q))

            goal_reg = layer['goal_reg'](mode_q)

        goal_FDE = self.goal_FDE(mode_q).squeeze(-1).T
        return mode_q, goal_reg, goal_FDE

    # ================================================================================
    # Initial Prediction (optimized)
    # ================================================================================
    def initial_prediction(self, mode_query, scene_context, bev_feat, goal_candidate, ego_dynamics):
        K, B, _ = mode_query.shape
        query_scale = self.get_query_scale_l1(mode_query)
        dec_embed = self.norm_l1[0](self.context_cross_attn_l1(query=mode_query, key=scene_context, 
                                                                     value=scene_context)[0] + mode_query)
        
        # coord transform of goal candidates (target-centric -> ego-centric)
        ego_loc, ego_sin, ego_cos = ego_dynamics['ego_loc'], ego_dynamics['ego_sin'], ego_dynamics['ego_cos']
        ego_loc, ego_sin, ego_cos = map(lambda t: t.unsqueeze(0).repeat(K, 1, 1), (ego_loc, ego_sin, ego_cos))
        
        rotation_matrix = torch.stack([
            torch.cat([ego_cos, -ego_sin], dim=-1),
            torch.cat([ego_sin, ego_cos], dim=-1)
        ], dim=-2)
        
        goal_candidate = goal_candidate - ego_loc
        goal_candidate = torch.matmul(goal_candidate.unsqueeze(-2), rotation_matrix).squeeze(-2)
        
        # cross attn with scene feature
        dec_embed = self.norm_l1[1](self.bev_cross_attn_l1(dec_embed=dec_embed, bev_feat=bev_feat,
                                                                 query_scale = query_scale, ref_points = goal_candidate))
        dec_embed = self.norm_l1[2](self.ffn_l1(dec_embed))
        
        dec_embed_T = self.tmp_MLP[0](dec_embed).reshape(K, B, self.T, -1)
        dec_embed_T = self.tmp_MLP[1](dec_embed_T)
        
        mode_prob = F.softmax(self.motion_cls_l1(dec_embed_T), dim=0).squeeze(dim=-1).T
        out_dist = self.motion_reg_l1(dec_embed_T)
        
        return dec_embed_T, mode_prob, out_dist
    # ================================================================================
    # Main Forward (unchanged behavior, optimized computation)
    # ================================================================================
    def forward(self, scene_context, bev_feat,
                ec_dyn, tc_dyn, ego_dyn, e2t_dict, **kwargs):

        B, _, _ = ec_dyn.shape
        n = scene_context.shape[1]

        # scene context reshape
        scene_context_repeat = scene_context.unsqueeze(2).repeat(1, 1, self.T, 1)
        scene_context_repeat = scene_context_repeat.permute(1, 0, 2, 3).reshape(n, B * self.T, -1)
        scene_context = scene_context.permute(1, 0, 2)

        # ego->target transform
        trans_x, trans_y = e2t_dict['trans_x'], e2t_dict['trans_y']
        rot_s, rot_c = e2t_dict['rot_sin'], e2t_dict['rot_cos']
        
        # -------------------- Goal Proposal BDA --------------------
        bda_sgcp_f, bda_sgcp_pos = self.bda_sgcp(bev_feat, ec_dyn, tc_dyn, e2t_dict)
        bda_sgcp_pos = ego_to_target(bda_sgcp_pos, trans_x, trans_y, rot_s, rot_c)

        mode_query, goal_reg, goal_FDE = self.goal_candidate_proposal(tc_dyn, bda_sgcp_f, bda_sgcp_pos)
        goal_candidate = goal_reg.detach()

        # -------------------- Initial Prediction --------------------
        dec_embed, init_mode_prob, init_pred_traj = self.initial_prediction(mode_query, scene_context, bev_feat, goal_candidate, ego_dyn)
        
        ref_points = init_pred_traj[..., :2].detach()
        mode_probs = [init_mode_prob]
        pred_trajs = [init_pred_traj.permute(0, 2, 1, 3)]
        
        dec_embed = dec_embed.permute(2, 1, 0, 3).reshape(self.T, B*self.K, -1)
        dec_embed = self.temp_pos_enc(dec_embed)
        for layer in self.dec_layers:
            query_scale = self.get_query_scale_T(dec_embed)
            dec_embed = layer(
                dec_embed=dec_embed,
                scene_context=scene_context_repeat,
                bev_feat=bev_feat,
                query_scale=query_scale,
                ref_points=ref_points,
                ego_dynamics=ego_dyn)
            mode_prob = F.softmax(self.motion_cls(dec_embed), dim=0).squeeze(dim=-1).T
            pred_traj = self.motion_reg(dec_embed)
            
            pred_traj[..., :2] += ref_points
            new_ref_points = pred_traj[..., :2]
            ref_points = new_ref_points.detach()
            pred_traj = pred_traj.permute(0, 2, 1, 3)
                
            mode_probs.append(mode_prob)
            pred_trajs.append(pred_traj)
            
            dec_embed = dec_embed.permute(2, 1, 0, 3).reshape(self.T, B*self.K, -1)
            
        output = {'predicted_probability': mode_probs,
                  'predicted_trajectory': pred_trajs,
                  'predicted_goal_FDE': goal_FDE,
                  'predicted_goal_reg': goal_reg}
        return output