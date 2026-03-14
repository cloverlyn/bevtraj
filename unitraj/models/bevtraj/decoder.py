import math
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from unitraj.models.bevtraj.bev_deformable_aggregation import BDA_DEC
from unitraj.models.bevtraj.decoder_deform_attn import BEVDeformCrossAttn
from unitraj.models.bevtraj.linear import MLP, FFN, MotionRegHead, MotionClsHead
from unitraj.models.bevtraj.utility import gen_sineembed_for_position, target_to_ego

from unitraj.models.bevtraj.temporal_attn import TemporalMHA, TemporalMHA_NoTimePE


# class TemporalPositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, future_len=12, temperature=500.0):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(future_len, d_model)
#         position = torch.arange(0, future_len).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, d_model, 2).float()
#                              * (-math.log(temperature) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer("pe", pe.unsqueeze(0))

#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1)]
#         return self.dropout(x)


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
        # self.temp_self_attn = nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout)

        # exp: temporal PE (time_embedding_mlp)
        self.temp_self_attn = TemporalMHA(self.D, self.num_heads, self.dropout)

        self.transformer_decoder_layer = nn.TransformerDecoderLayer(self.D, self.num_heads,
                                                                    dim_feedforward=self.ffn_D, dropout=self.dropout)
        self.bev_cross_attn = BEVDeformCrossAttn(**config['deform_cross_attn'])

        # hybrid self-attn: token 길이를 K*T로 보고 attention
        self.hybrid_self_attn = nn.MultiheadAttention(
            self.D, self.num_heads, dropout=self.dropout
        )

        self.ffn = FFN(self.D, self.ffn_D, 2)
    
    def forward(self, dec_embed, scene_context, bev_feat, query_scale, ref_points, ego_dyn, time_pe):
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
        
        # dec_embed = self.norm[0](self.temp_self_attn(query=dec_embed, key=dec_embed, value=dec_embed)[0] + dec_embed)

        # exp: temporal PE (time_embedding_mlp)
        temp_out = self.temp_self_attn(dec_embed, time_pe)
        dec_embed = self.norm[0](temp_out + dec_embed)
        
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

        # 5) hybrid self-attn on K*T tokens
        # hybrid_tokens = dec_embed.permute(0, 2, 1, 3).reshape(self.K * self.T, B, self.D)  # [K*T,B,D]
        # hybrid_out = self.hybrid_self_attn(
            # query=hybrid_tokens, key=hybrid_tokens, value=hybrid_tokens
        # )[0]
        # hybrid_tokens = self.norm[2](hybrid_out + hybrid_tokens)
        # restore [K,B,T,D]
        # dec_embed = hybrid_tokens.reshape(self.K, self.T, B, self.D).permute(0, 2, 1, 3).contiguous()
        # =================

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
        self.grid_size = config['grid_size']
        
        self.dca_cfg = config['deform_cross_attn']
        self.dca_cfg['dim'] = self.D

        self.dca_itr_cfg = config['deform_cross_attn_itr']
        self.dca_itr_cfg['dim'] = self.D

        self.dec_layer_config = {
            'future_len': self.T,
            'd_model': self.D,
            'query_dims': self.query_scale_dims,
            'ffn_dims': self.ffn_D,
            'spa_pos_T': self.spa_pos_T,
            'num_modes': self.K,
            'dropout': self.dropout,
            'num_heads': self.num_heads,
            'deform_cross_attn': self.dca_itr_cfg,
        }
        
        # ============================ Goal Candidate Proposal==========================

        # classification
        self.bda_sgcp = BDA_DEC(self.config['bda_dec'], self.D)
        self.goal_prob = MLP(self.D, self.D, 1, 2)
        # self.goal_FDE = MLP(self.D, self.D, 1, 2)

        # regression
        self.goal_proposal = []
        for _ in range(self.L_goal_proposal):
            goal_proposal_layer = nn.ModuleDict({
                'deform_cross_attn': BEVDeformCrossAttn(**self.dca_cfg),
                'norm1': nn.LayerNorm(self.D),
                'mode_self_attn': nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout),
                'norm2': nn.LayerNorm(self.D),
                'ffn': FFN(self.D, self.ffn_D, 2),
                'norm3': nn.LayerNorm(self.D),
            })
            self.goal_proposal.append(goal_proposal_layer)
        self.goal_proposal = nn.ModuleList(self.goal_proposal)

        self.get_query_scale_sgcp = MLP(self.D, self.query_scale_dims, self.query_scale_dims, 2)
        self.goal_reg = MLP(self.D, self.D, 2, 2)
        # self.goal_FDE = MLP(self.D, self.D, 1, 2)
        self.goal_prob_reg = MLP(self.D, self.D, 1, 2)

        self.register_buffer('denorm_scale', torch.tensor(self.grid_size, dtype=torch.float32))

        # ============================ Initial Prediction ==============================
        self.get_query_scale_itp = MLP(self.D, self.query_scale_dims, self.query_scale_dims, 2)
        self.norm_l1 = nn.ModuleList([nn.LayerNorm(self.D) for _ in range(3)])
        
        self.context_cross_attn_l1 = nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout)
        self.bev_cross_attn_l1 = BEVDeformCrossAttn(**self.dca_cfg)
        self.ffn_l1 = FFN(self.D, self.ffn_D, 2)
        
        # self.tmp_MLP = nn.ModuleList([
        #     nn.Sequential(nn.Linear(self.D, self.T_D * self.T), nn.GELU()),
        #     nn.Sequential(nn.Linear(self.T_D, self.D), nn.GELU())
        # ])
        self.motion_cls_l1 = MotionClsHead(self.D, self.T_D, self.T)
        self.motion_reg_l1 = MotionRegHead(self.D)

        # exp: DeMo-like ITP (state consistency branch)
        self.state_norm_l1 = nn.ModuleList([nn.LayerNorm(self.D) for _ in range(2)])
        self.state_context_cross_attn_l1 = nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout)
        self.state_temp_self_attn_l1 = TemporalMHA_NoTimePE(self.D, self.num_heads, self.dropout)
        # state query auxiliary prediction head (B,T,2 supervision)
        self.state_reg_l1 = MLP(self.D, self.D, 2, 2)


        # ============================ Iterative Refinement ============================

        # exp: temporal PE (time_embedding_mlp)
        self.time_embedding_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, self.D)
        )
        self.register_buffer("future_time", torch.arange(self.T).float().unsqueeze(-1))
        self.dt = config.get("dt", 0.1)
        self.time_emb_alpha = nn.Parameter(torch.tensor(1.0))
        
        # self.mode_sep_enc = ModeSeperationEncoding(self.D, self.dropout, mode_num=self.K, temperature=self.mode_pos_T)
        self.get_query_scale_itr = MLP(self.query_scale_dims, self.query_scale_dims, self.query_scale_dims, 2)
        
        dec_layer = BEVTrajDecoderLayer(self.dec_layer_config)
        self.dec_layers = nn.ModuleList([copy.deepcopy(dec_layer) for _ in range(self.L_dec - 1)])
        
        self.motion_cls = MotionClsHead(self.D, self.T_D, self.T)
        self.motion_reg = MotionRegHead(self.D)
        self.motion_reg_final = MotionRegHead(self.D)

        # exp: sample-conditioned deterministic code
        # self.temp_pos_enc = TemporalPositionalEncoding(self.D, self.dropout, future_len=self.T, temperature=10000)

    def build_time_pe(self, B, K, dtype):
        t = self.future_time * self.dt + 0.1
        t = t.to(dtype) # [T,1]
        pe = self.time_embedding_mlp(t)  # [T,D]
        pe = pe[:, None, None, :].repeat(1, B, K, 1)  # [T,B,K,D]
        pe = pe.reshape(self.T, B * K, self.D)  # [T,BK,D]

        return self.time_emb_alpha * pe

    def goal_candidate_proposal(self, bev_feat, ec_dyn, tc_dyn, ego_dyn):

        # =============================== classification ===============================

        bda_token, bda_pos = self.bda_sgcp(bev_feat, ec_dyn, tc_dyn, ego_dyn)

        B, _, D = bda_token.shape

        goal_logit = self.goal_prob(bda_token).squeeze(-1)
        goal_prob  = F.softmax(goal_logit, dim=-1)

        # top-k
        num_goal = 64 # hard code
        # _, top_idx = torch.topk(goal_prob, k=self.K, dim=-1)
        _, top_idx = torch.topk(goal_prob, k=num_goal, dim=-1)

        # idx_tok = top_idx.unsqueeze(-1).expand(B, self.K, D)
        # idx_pos = top_idx.unsqueeze(-1).expand(B, self.K, 2)
        idx_tok = top_idx.unsqueeze(-1).expand(B, num_goal, D)
        idx_pos = top_idx.unsqueeze(-1).expand(B, num_goal, 2)

        mode_query = torch.gather(bda_token, dim=1, index=idx_tok).permute(1, 0, 2).contiguous()
        goal_pos = torch.gather(bda_pos, dim=1, index=idx_pos)

        # ================================= regression =================================

        ref_pos = goal_pos.detach()

        trans_x, trans_y, rot_sin, rot_cos = (
            ego_dyn['ego_x'],
            ego_dyn['ego_y'],
            ego_dyn['ego_sin'],
            ego_dyn['ego_cos'],
        )

        goal_reg_list = []
        goal_prob_list = []
        goal_prob_list.append(goal_prob)

        for lid, layer in enumerate(self.goal_proposal):
            query_scale = self.get_query_scale_sgcp(mode_query)
            ref_pos_ego = target_to_ego(ref_pos, trans_x, trans_y, rot_sin, rot_cos).permute(1, 0, 2)

            mode_query = layer['norm1'](layer['deform_cross_attn']( \
                    dec_embed = mode_query, bev_feat = bev_feat, query_scale = query_scale, ref_points = ref_pos_ego))
            mode_query = layer['norm2'](layer['mode_self_attn']( \
                    query = mode_query, key = mode_query, value = mode_query)[0] + mode_query)
            mode_query = layer['norm3'](layer['ffn'](mode_query))

            tmp = self.goal_reg(mode_query)
            goal_reg = tmp + ref_pos.permute(1, 0, 2)

            goal_reg_list.append(goal_reg)
            ref_pos = goal_reg.detach().permute(1, 0, 2)

            goal_logit = self.goal_prob_reg(mode_query).squeeze(-1).T
            goal_prob = F.softmax(goal_logit, dim=-1)
            goal_prob_list.append(goal_prob)
            
        # return mode_query, goal_prob, bda_pos, goal_reg_list, goal_FDE_list
        return mode_query, bda_pos, goal_reg_list, goal_prob_list

    def initial_prediction(self, mode_query, scene_context, bev_feat, goal_candidate, ego_dyn):
        """
        mode_query:    [M, B, D]   (M = 64)
        scene_context: [n, B, D]
        """
        M, B, _ = mode_query.shape
        K = self.K  # 최종 사용할 mode 수 (10)

        # ===================== mode localization branch =====================
        query_scale = self.get_query_scale_itp(mode_query)

        mode_embed = self.norm_l1[0](
            self.context_cross_attn_l1(
                query=mode_query, key=scene_context, value=scene_context
            )[0] + mode_query
        )

        trans_x, trans_y, rot_sin, rot_cos = (
            ego_dyn['ego_x'],
            ego_dyn['ego_y'],
            ego_dyn['ego_sin'],
            ego_dyn['ego_cos'],
        )

        goal_candidate_target = goal_candidate.permute(1, 0, 2)  # [B, M, 2] (target coord)
        goal_candidate = target_to_ego(
            goal_candidate_target, trans_x, trans_y, rot_sin, rot_cos
        ).permute(1, 0, 2)  # [M, B, 2] (ego coord for BEV cross-attn)

        mode_embed = self.norm_l1[1](
            self.bev_cross_attn_l1(
                dec_embed=mode_embed,
                bev_feat=bev_feat,
                query_scale=query_scale,
                ref_points=goal_candidate,
            )
        )

        mode_embed = self.norm_l1[2](self.ffn_l1(mode_embed))  # [M,B,D]

        # ===================== state consistency branch =====================
        t = (self.future_time * self.dt + 0.1).to(
            device=mode_query.device, dtype=mode_query.dtype
        )

        state_query = self.time_emb_alpha * self.time_embedding_mlp(t)  # [T,D]
        state_query = state_query.unsqueeze(1).repeat(1, B, 1)  # [T,B,D]

        state_query = self.state_norm_l1[0](
            self.state_context_cross_attn_l1(
                query=state_query, key=scene_context, value=scene_context
            )[0] + state_query
        )

        state_query = self.state_norm_l1[1](
            self.state_temp_self_attn_l1(state_query, None) + state_query
        )

        state_pred = self.state_reg_l1(state_query).permute(1, 0, 2).contiguous()

        # ===================== hybrid coupling =====================
        mode_bt = mode_embed.permute(1, 0, 2).unsqueeze(2)  # [B,M,1,D]
        state_bt = state_query.permute(1, 0, 2).unsqueeze(1)  # [B,1,T,D]

        dec_embed_T = mode_bt + state_bt  # [B,M,T,D]
        dec_embed_T = dec_embed_T.permute(1, 0, 2, 3).contiguous()  # [M,B,T,D]

        # ===================== trajectory prediction =====================
        mode_prob = self.motion_cls_l1(dec_embed_T).squeeze(dim=-1).T  # [B,M]
        out_dist = self.motion_reg_l1(dec_embed_T)  # [M,B,T,5]

        # # ===================== top-K selection =====================
        # _, top_idx = torch.topk(mode_prob, K, dim=1)  # [B,K]

        # # gather indices
        # idx_mode = top_idx.permute(1,0)  # [K,B]

        # idx_embed = idx_mode.unsqueeze(-1).unsqueeze(-1).expand(K, B, self.T, self.D)
        # dec_embed_T = torch.gather(dec_embed_T, dim=0, index=idx_embed)

        # idx_traj = idx_mode.unsqueeze(-1).unsqueeze(-1).expand(K, B, self.T, 5)
        # out_dist_K = torch.gather(out_dist, dim=0, index=idx_traj)

        # # goal candidates in target coord selected by initial_prediction top-k
        # idx_goal = top_idx.unsqueeze(-1).expand(B, K, 2)
        # goal_candidate_topk = torch.gather(goal_candidate_target, dim=1, index=idx_goal)  # [B, K, 2]

        return dec_embed_T, mode_prob, out_dist, state_pred

    def forward(self, scene_context, bev_feat, ec_dyn, tc_dyn, ego_dyn, **kwargs):

        B, _, _ = ec_dyn.shape
        n = scene_context.shape[1]

        scene_context_repeat = scene_context.unsqueeze(2).repeat(1, 1, self.T, 1)
        scene_context_repeat = scene_context_repeat.permute(1, 0, 2, 3).reshape(n, B * self.T, -1)
        scene_context = scene_context.permute(1, 0, 2)

        # -------------------Goal Candidate Proposal -----------------
        mode_query, bda_pos, goal_reg_list, goal_prob_list = \
            self.goal_candidate_proposal(
                bev_feat, ec_dyn, tc_dyn, ego_dyn
            )
        goal_candidate = goal_reg_list[-1].detach()

        # -------------------- Initial Prediction --------------------
        dec_embed, init_mode_prob, init_pred_traj, state_pred = \
            self.initial_prediction(mode_query, scene_context, bev_feat, goal_candidate, ego_dyn)
        
        # ref_points = init_pred_traj[..., :2].detach().clone()
        ref_points = init_pred_traj[..., :2].detach().clone()
        mode_probs = [init_mode_prob]
        pred_trajs = [init_pred_traj.permute(0, 2, 1, 3)]
        
        dec_embed = dec_embed.permute(2, 1, 0, 3).reshape(self.T, B * self.K, -1)

        # exp: sample-conditioned deterministic code
        # dec_embed = self.temp_pos_enc(dec_embed)

        # exp: temporal PE (time_embedding_mlp)
        time_pe = self.build_time_pe(B, self.K, dec_embed.dtype)

        num_refine_layers = len(self.dec_layers)
        for lid, layer in enumerate(self.dec_layers):
            # is_last_layer = (lid == num_refine_layers - 1)
            query_scale = self.get_query_scale_itr(dec_embed)
            dec_embed = layer(
                dec_embed=dec_embed,
                scene_context=scene_context_repeat,
                bev_feat=bev_feat,
                query_scale=query_scale,
                ref_points=ref_points,
                ego_dyn=ego_dyn,
                time_pe=time_pe,
                )
            
            # mode_prob = F.softmax(self.motion_cls(dec_embed), dim=0).squeeze(dim=-1).T
            mode_prob = self.motion_cls(dec_embed).squeeze(dim=-1).T

            # if is_last_layer:
            #     # final layer: direct prediction (no refinement)
            #     pred_traj = self.motion_reg_final(dec_embed)
            # else:
                # intermediate layers: iterative refinement
            pred_traj_raw = self.motion_reg(dec_embed)          # [K, B, T, 5]
            pred_xy = pred_traj_raw[..., :2] + ref_points       # out-of-place
            pred_traj = torch.cat([pred_xy, pred_traj_raw[..., 2:]], dim=-1)
            ref_points = pred_xy.detach().clone()

            # pred_traj = self.motion_reg(dec_embed)          # [K, B, T, 5]
            # ref_points = pred_traj.detach().clone()[..., :2]

            pred_traj = pred_traj.permute(0, 2, 1, 3)
                
            mode_probs.append(mode_prob)
            pred_trajs.append(pred_traj)
            
            dec_embed = dec_embed.permute(2, 1, 0, 3).reshape(self.T, B*self.K, -1)
            
        output = {'predicted_probability': mode_probs,
                  'predicted_trajectory': pred_trajs,
                  'anchor_pos' : bda_pos,
                  'goal_reg_list': goal_reg_list,
                  'goal_prob_list' : goal_prob_list,
                #   'init_top_idx': init_top_idx,                # [B, K]
                #   'goal_candidate_topk': goal_candidate_topk,  # [B, K, 2]
                  'state_pred': state_pred, # [B, T, 2]
                #   'goal_FDE_list': goal_FDE_list,
                }
        return output
    
