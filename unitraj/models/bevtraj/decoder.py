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
        
        # ============================ BDA modules ============================
        self.bda_sgcp = BDA_DEC(self.config['bda_dec'], self.D)
        self.goal_score_head = MLP(self.D, self.D, 1, 2)

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
        
        # self.mode_sep_enc = ModeSeperationEncoding(self.D, self.dropout, mode_num=self.K, temperature=self.mode_pos_T)
        self.get_query_scale_T = MLP(self.query_scale_dims, self.query_scale_dims, self.query_scale_dims, 2)
        
        dec_layer = BEVTrajDecoderLayer(self.dec_layer_config)
        self.dec_layers = nn.ModuleList([copy.deepcopy(dec_layer) for _ in range(self.L_dec - 1)])
        
        self.motion_cls = MotionClsHead(self.D, self.T_D, self.T)
        self.motion_reg = MotionRegHead(self.D)

    # def goal_candidate_proposal(self, bev_feat, ec_dyn, tc_dyn, ego_dyn):
    #     bda_token, bda_pos = self.bda_sgcp(bev_feat, ec_dyn, tc_dyn, ego_dyn)

    #     goal_logit = self.goal_score_head(bda_token).squeeze(-1)
    #     goal_prob = F.softmax(goal_logit, dim=-1) # (B, N)



    #     return mode_query, goal_candidate

    def batched_kmeans_topk(
            self,
            top_pos: torch.Tensor,   # [B, M, 2]
            top_token: torch.Tensor, # [B, M, D]
            top_weight: torch.Tensor,# [B, M]
            K: int,
            iters: int = 10,
            eps: float = 1e-6,
        ):
        """
        Returns:
            mode_query:     [K, B, D]
            goal_candidate: [B, K, 2]
        """

        B, M, _ = top_pos.shape
        D = top_token.shape[-1]

        pts = top_pos.detach()  # clustering은 gradient 불필요

        # ---------------- init centers ----------------
        if M >= K:
            centers = pts[:, :K, :].clone()  # [B, K, 2]
        else:
            pad = pts[:, :1, :].expand(B, K - M, 2)
            centers = torch.cat([pts, pad], dim=1).contiguous()

        # ---------------- kmeans loop ----------------
        for _ in range(iters):
            dist = (pts.unsqueeze(2) - centers.unsqueeze(1)).pow(2).sum(-1)  # [B, M, K]
            labels = dist.argmin(dim=-1)                                      # [B, M]

            new_centers = centers.clone()

            for k in range(K):
                mask = (labels == k)                          # [B, M]
                cnt = mask.sum(dim=-1, keepdim=True)          # [B, 1]
                has = (cnt.squeeze(-1) > 0)

                if has.any():
                    m = mask.unsqueeze(-1).float()            # [B, M, 1]
                    sum_xy = (pts * m).sum(dim=1)             # [B, 2]
                    mean_xy = sum_xy / (cnt.float() + eps)    # [B, 2]
                    new_centers[has, k, :] = mean_xy[has]

            centers = new_centers

        goal_candidate = centers  # [B, K, 2]

        # ---------------- mode query ----------------
        dist = (pts.unsqueeze(2) - goal_candidate.unsqueeze(1)).pow(2).sum(-1)
        labels = dist.argmin(dim=-1)  # [B, M]

        mode_query_BKD = top_token.new_zeros(B, K, D)

        for k in range(K):
            mask = (labels == k).float()            # [B, M]
            w = top_weight * mask                   # [B, M]
            denom = w.sum(dim=-1, keepdim=True) + eps
            w_norm = (w / denom).unsqueeze(-1)      # [B, M, 1]
            mode_query_BKD[:, k, :] = (top_token * w_norm).sum(dim=1)

        mode_query = mode_query_BKD.permute(1, 0, 2).contiguous()  # [K, B, D]

        return mode_query, goal_candidate

    def goal_candidate_proposal(self, bev_feat, ec_dyn, tc_dyn, ego_dyn):
        bda_token, bda_pos = self.bda_sgcp(bev_feat, ec_dyn, tc_dyn, ego_dyn)

        B, N, D = bda_token.shape
        topM = min(64, N)

        goal_logit = self.goal_score_head(bda_token).squeeze(-1)
        goal_prob  = F.softmax(goal_logit, dim=-1)

        # top-k
        top_prob, top_idx = torch.topk(goal_prob, k=topM, dim=-1)

        idx_tok = top_idx.unsqueeze(-1).expand(B, topM, D)
        idx_pos = top_idx.unsqueeze(-1).expand(B, topM, 2)

        top_token = torch.gather(bda_token, dim=1, index=idx_tok)
        top_pos   = torch.gather(bda_pos,   dim=1, index=idx_pos)

        mode_query, goal_candidate = self.batched_kmeans_topk(
            top_pos=top_pos,
            top_token=top_token,
            top_weight=top_prob,
            K=self.K,
            iters=10,
        )

        return mode_query, goal_candidate, goal_prob, bda_pos

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
        # goal_candidate = goal_candidate.permute(1, 0, 2)
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

        # -------------------- Goal Proposal ------------------------
        # bda_token, bda_pos = self.bda_sgcp(bev_feat, ec_dyn, tc_dyn, ego_dyn)
        # goal_logit = self.goal_score_head(bda_token).squeeze(-1) # (B, N)

        # trans_x, trans_y, rot_sin, rot_cos = (
        #     ego_dyn['ego_x'],
        #     ego_dyn['ego_y'],
        #     ego_dyn['ego_sin'],
        #     ego_dyn['ego_cos'],
        # )
        # bda_pos = ego_to_target(bda_pos, trans_x, trans_y, rot_sin, rot_cos)

        # goal_logit = self.goal_score_head(bda_token).squeeze(-1) # (B, N)

        # qqq = self.Q.repeat(1, B, 1) # (K, B, D)

        # goal_candidate = goal_reg.detach()

        mode_query, goal_candidate, goal_prob, bda_pos = self.goal_candidate_proposal(bev_feat, ec_dyn, tc_dyn, ego_dyn)

        # -------------------- Initial Prediction --------------------

        dec_embed, init_mode_prob, init_pred_traj = self.initial_prediction(mode_query, scene_context, bev_feat, goal_candidate, ego_dyn)
        
        ref_points = init_pred_traj[..., :2].detach()
        mode_probs = [init_mode_prob]
        pred_trajs = [init_pred_traj.permute(0, 2, 1, 3)]
        
        # mode seperation encoding
        # dec_embed = dec_embed.permute(2, 1, 0, 3) # (T, B, K, -1)
        # dec_embed = self.mode_sep_enc(dec_embed).reshape(self.T, B * self.K, -1)
        dec_embed = dec_embed.permute(2, 1, 0, 3).reshape(self.T, B * self.K, -1)

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
                #   'predicted_goal_FDE': goal_FDE,
                #   'predicted_goal_reg': goal_reg
                  'goal_prob' : goal_prob,
                  'anchor_pos' : bda_pos,
                  'goal_candidate': goal_candidate,
                }
        return output
    