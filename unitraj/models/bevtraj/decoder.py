import math
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from unitraj.models.bevtraj.bev_deformable_aggregation import BDA_DEC
from unitraj.models.bevtraj.decoder_deform_attn import BEVDeformCrossAttn
from unitraj.models.bevtraj.linear import MLP, FFN, MotionRegHead, MotionClsHead
from unitraj.models.bevtraj.utility import gen_sineembed_for_position, ego_to_target, target_to_ego

    
class ModeSeperationEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, mode_num=10, temperature=500.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(mode_num, d_model)
        position = torch.arange(0, mode_num).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(temperature) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)


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
        self.bev_cross_attn = BEVDeformCrossAttn(**config['deform_cross_attn'])
        self.ffn = FFN(self.D, self.ffn_D, 2)
    
    def forward(self, dec_embed, scene_context, bev_feat, query_scale, ref_points, ego_dyn):
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
        dec_embed = self.transformer_decoder_layer(tgt=dec_embed.reshape(self.K, B * self.T, -1),
                                                   memory=scene_context).reshape(self.K, B, self.T, -1)
        
        # ============================== ego-centric(ec) modeling ==============================
        trans_x, trans_y, rot_sin, rot_cos = (
            ego_dyn['ego_x'],
            ego_dyn['ego_y'],
            ego_dyn['ego_sin'],
            ego_dyn['ego_cos'],
        )
        ref_points_flat = ref_points.permute(1, 0, 2, 3).reshape(B, self.K * self.T, 2)
        ref_points_flat = target_to_ego(ref_points_flat, trans_x, trans_y, rot_sin, rot_cos)
        ref_points = ref_points_flat.reshape(B, self.K, self.T, 2).permute(1, 0, 2, 3)
        
        # cross attn with bev feature
        dec_embed = self.norm[1](self.bev_cross_attn(dec_embed, bev_feat, query_scale, ref_points))
        dec_embed = self.norm[2](self.ffn(dec_embed))
        
        return dec_embed


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
        self.mode_pos_T = config['mode_pos_T']
        self.spa_pos_T = config['spa_pos_T']
        self.dropout = config['dropout']
        self.L_goal_proposal = config['num_goal_proposal_layers']
        self.L_dec = config['num_decoder_layers']
        self.num_heads = config['num_heads']
        
        self.dca_cfg = config['deform_cross_attn']
        self.dca_cfg['dim'] = self.D
        
        self.dec_layer_config = {
            'future_len': self.T,
            'd_model': self.D,
            'query_dims': self.query_scale_dims,
            'ffn_dims': self.ffn_D,
            'spa_pos_T': self.spa_pos_T,
            'num_modes': self.K,
            'dropout': self.dropout,
            'num_heads': self.num_heads,
            'deform_cross_attn': self.dca_cfg,
        }
        
        # ============================ Learnable mode queries ============================
        self.Q = nn.Parameter(torch.empty(self.K, 1, self.D))
        nn.init.xavier_uniform_(self.Q)

        # ============================ BDA modules ============================
        self.bda_sgcp = BDA_DEC(self.config['bda_dec'], self.D)

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

        self.grid_size = config['grid_size']
        self.register_buffer('denorm_scale', torch.tensor(self.grid_size, dtype=torch.float32))

        # ============================ Initial Prediction ============================
        self.get_query_scale_l1 = MLP(self.D, self.query_scale_dims, self.query_scale_dims, 2)
        self.norm_l1 = nn.ModuleList([nn.LayerNorm(self.D) for _ in range(3)])
        
        self.context_cross_attn_l1 = nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout)
        self.bev_cross_attn_l1 = BEVDeformCrossAttn(**self.dca_cfg)
        self.ffn_l1 = FFN(self.D, self.ffn_D, 2)
        
        self.tmp_MLP = nn.ModuleList([
            nn.Sequential(nn.Linear(self.D, self.T_D * self.T), nn.GELU()),
            nn.Sequential(nn.Linear(self.T_D, self.D), nn.GELU())
        ])
        self.motion_cls_l1 = MotionClsHead(self.D, self.T_D, self.T)
        self.motion_reg_l1 = MotionRegHead(self.D)

        # ============================ Iterative Refinement ============================
        
        self.mode_sep_enc = ModeSeperationEncoding(self.D, self.dropout, mode_num=self.K, temperature=self.mode_pos_T)
        self.get_query_scale_T = MLP(self.query_scale_dims, self.query_scale_dims, self.query_scale_dims, 2)
        
        dec_layer = BEVTrajDecoderLayer(self.dec_layer_config)
        self.dec_layers = nn.ModuleList([copy.deepcopy(dec_layer) for _ in range(self.L_dec - 1)])
        
        self.motion_cls = MotionClsHead(self.D, self.T_D, self.T)
        self.motion_reg = MotionRegHead(self.D)
        
    def goal_candidate_proposal(self, tc_dyn, bda_token, bda_pos):
        B = tc_dyn.size(0)

        # dynamic feature encoding (query positional encoding)
        d = self.tc_dynamic_encoder['enc'](tc_dyn).reshape(B, -1)
        d_s = self.tc_dynamic_encoder['enc_s'](d).unsqueeze(0).expand(self.K, -1, -1)
        d_s = self.tc_dynamic_encoder['enc_s_mode'](torch.cat([self.Q.expand(-1, B, -1), d_s], dim=-1))
        d_c = self.tc_dynamic_encoder['enc_c'](d).unsqueeze(0).expand(self.K, -1, -1)
        d_c = self.tc_dynamic_encoder['enc_c_mode'](torch.cat([self.Q.expand(-1, B, -1), d_c], dim=-1))

        temperature = self.config['spa_pos_T']
        pos_k = gen_sineembed_for_position(bda_pos, hidden_dim=self.D, temperature=temperature)
        k = torch.cat([bda_token, pos_k], dim=-1).permute(1, 0, 2)
        v = bda_token.permute(1, 0, 2)

        goal_reg_list = []
        goal_FDE_list = []
        for i, layer in enumerate(self.goal_proposal):
            if i==0:
                src = self.Q.expand(-1, B, -1)
                mode_q = self.Q.expand(-1, B, -1) + d_s
                
            else:
                src = mode_q
                goal_reg_pos = gen_sineembed_for_position(goal_reg, hidden_dim=self.D, temperature=temperature)
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

            # goal_reg = layer['goal_reg'](mode_q)
            if i==0:
                goal_reg = layer['goal_reg'](mode_q).tanh() * self.denorm_scale[None, None, :]
            else:
                tmp = layer['goal_reg'](mode_q) * 3.0    # hard code
                goal_reg = ref_pos + tmp

            goal_reg_list.append(goal_reg)
            ref_pos = goal_reg.detach()

            goal_FDE = self.goal_FDE(mode_q).squeeze(-1).T
            goal_FDE_list.append(goal_FDE)

        return mode_q, goal_reg_list, goal_FDE_list

    def initial_prediction(self, mode_query, scene_context, bev_feat, goal_candidate, ego_dyn):
        K, B, _ = mode_query.shape
        query_scale = self.get_query_scale_l1(mode_query)
        dec_embed = self.norm_l1[0](self.context_cross_attn_l1(query=mode_query, key=scene_context, 
                                                                     value=scene_context)[0] + mode_query)
        
        trans_x, trans_y, rot_sin, rot_cos = (
            ego_dyn['ego_x'],
            ego_dyn['ego_y'],
            ego_dyn['ego_sin'],
            ego_dyn['ego_cos'],
        )
        goal_candidate = goal_candidate.permute(1, 0, 2)
        goal_candidate = target_to_ego(goal_candidate, trans_x, trans_y, rot_sin, rot_cos).permute(1, 0, 2)
        
        # cross attn with scene feature
        dec_embed = self.norm_l1[1](self.bev_cross_attn_l1(dec_embed=dec_embed, bev_feat=bev_feat,
                                                                 query_scale = query_scale, ref_points = goal_candidate))
        dec_embed = self.norm_l1[2](self.ffn_l1(dec_embed))
        
        dec_embed_T = self.tmp_MLP[0](dec_embed).reshape(K, B, self.T, -1)
        dec_embed_T = self.tmp_MLP[1](dec_embed_T)
        
        mode_prob = F.softmax(self.motion_cls_l1(dec_embed_T), dim=0).squeeze(dim=-1).T
        out_dist = self.motion_reg_l1(dec_embed_T)
        
        return dec_embed_T, mode_prob, out_dist

    def forward(self, scene_context, bev_feat, ec_dyn, tc_dyn, ego_dyn, **kwargs):

        B, _, _ = ec_dyn.shape
        n = scene_context.shape[1]

        scene_context_repeat = scene_context.unsqueeze(2).repeat(1, 1, self.T, 1)
        scene_context_repeat = scene_context_repeat.permute(1, 0, 2, 3).reshape(n, B * self.T, -1)
        scene_context = scene_context.permute(1, 0, 2)

        # -------------------- Goal Proposal BDA --------------------
        bda_token, bda_sgcp_pos = self.bda_sgcp(bev_feat, ec_dyn, tc_dyn)

        trans_x, trans_y, rot_sin, rot_cos = (
            ego_dyn['ego_x'],
            ego_dyn['ego_y'],
            ego_dyn['ego_sin'],
            ego_dyn['ego_cos'],
        )
        bda_sgcp_pos = ego_to_target(bda_sgcp_pos, trans_x, trans_y, rot_sin, rot_cos)

        mode_query, goal_reg_list, goal_FDE_list = self.goal_candidate_proposal(tc_dyn, bda_token, bda_sgcp_pos)
        goal_candidate = goal_reg_list[-1].detach()

        # -------------------- Initial Prediction --------------------
        dec_embed, init_mode_prob, init_pred_traj = self.initial_prediction(mode_query, scene_context, bev_feat, goal_candidate, ego_dyn)
        
        ref_points = init_pred_traj[..., :2].detach()
        mode_probs = [init_mode_prob]
        pred_trajs = [init_pred_traj.permute(0, 2, 1, 3)]
        
        # mode seperation encoding
        dec_embed = dec_embed.permute(2, 1, 0, 3) # (T, B, K, -1)
        dec_embed = self.mode_sep_enc(dec_embed).reshape(self.T, B * self.K, -1)

        for layer in self.dec_layers:
            query_scale = self.get_query_scale_T(dec_embed)
            dec_embed = layer(
                dec_embed=dec_embed,
                scene_context=scene_context_repeat,
                bev_feat=bev_feat,
                query_scale=query_scale,
                ref_points=ref_points,
                ego_dyn=ego_dyn)
            
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
                  'predicted_goal_FDE': goal_FDE_list,
                  'predicted_goal_reg': goal_reg_list}
        return output
    