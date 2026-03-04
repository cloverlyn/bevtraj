import torch
import torch.nn as nn
import torch.nn.functional as F


class Criterion(nn.Module):
    def __init__(self, config):
        super(Criterion, self).__init__()
        self.config = config

        self.goal_FDE_loss = nn.SmoothL1Loss(reduction='mean', beta=1.0)

    def forward(self, out, gt, center_gt_final_valid_idx, traj_data):
        modes_preds = out['predicted_probability'] # [B, K]
        preds = out['predicted_trajectory'] # [K, T, B, 5]
        # goal_prob = out['goal_prob']
        goal_FDE = out['goal_FDE']
        anchor_pos = out['anchor_pos']
        goal_candidate = out['goal_candidate']       # [B, K, 2]
        dense_future_pred = out['dense_future_pred']

        gt_decoder = gt[0]
        gt_dense_future_trajs = gt[1]
        
        # iterative decoder loss: SGCP hard assignment
        decoder_loss = self.get_decoder_loss_hard_assign(
            modes_preds=modes_preds,
            preds=preds,
            goal_candidate=goal_candidate,
            gt_decoder=gt_decoder,
            center_gt_final_valid_idx=center_gt_final_valid_idx,
        )

        # goal_prob_loss = self.get_goal_prob_loss(
        #     goal_prob=goal_prob,
        #     goal_anchor=anchor_pos,
        #     gt=gt_decoder,
        #     center_gt_final_valid_idx=center_gt_final_valid_idx,
        # )
        goal_FDE_loss = self.get_goal_FDE_loss(
            anchor_pos=anchor_pos,
            goal_FDE=goal_FDE,
            gt_decoder=gt_decoder,
            center_gt_final_valid_idx=center_gt_final_valid_idx,
        )

        dense_future_loss = self.get_dense_future_prediction_loss(dense_future_pred, gt_dense_future_trajs)

        # total_loss = decoder_loss + goal_prob_loss + dense_future_loss
        total_loss = decoder_loss + goal_FDE_loss + dense_future_loss
        return total_loss

    def get_decoder_loss_hard_assign(
        self,
        modes_preds,                 # list of [B, K]
        preds,                       # list of [K, T, B, 5]
        goal_candidate,              # [B, K, 2] (SGCP top-k candidate)
        gt_decoder,                  # [B, T, 3] -> (x, y, valid)
        center_gt_final_valid_idx,   # [B]
    ):
        device = gt_decoder.device
        B = gt_decoder.size(0)
        b_idx = torch.arange(B, device=device)

        gt_xy = gt_decoder[..., :2]                     # [B, T, 2]
        gt_mask = gt_decoder[..., 2].float()            # [B, T]
        final_idx = center_gt_final_valid_idx.long()    # [B]
        gt_goal = gt_xy[b_idx, final_idx]               # [B, 2]
        valid_final = gt_mask[b_idx, final_idx]         # [B]

        # 1) hard assignment from SGCP goal candidate (detach)
        with torch.no_grad():
            dist = (goal_candidate.detach() - gt_goal[:, None, :]).norm(dim=-1)  # [B, K]
            hard_idx = dist.argmin(dim=-1)                                        # [B]

        w_cls = self.config.get('cls_weight', 1.0)
        w_reg = self.config.get('reg_weight', 1.0)

        total = 0.0
        for pred_scores, pred in zip(modes_preds, preds):
            # pred: [K, T, B, 5] -> [B, K, T, 5]
            pred_trajs = pred.permute(2, 0, 1, 3).contiguous()

            # MTR nll_loss_gmm_direct expects log_std, but MotionRegHead outputs sigma
            mu = pred_trajs[..., :2]
            log_std = torch.log(pred_trajs[..., 2:4].clamp_min(1e-6))
            rho = pred_trajs[..., 4:5]
            pred_trajs_gmm = torch.cat([mu, log_std, rho], dim=-1)

            loss_reg_gmm, hard_idx = self.nll_loss_gmm_direct(
                pred_scores=pred_scores,                 # [B, K]
                pred_trajs=pred_trajs_gmm,               # [B, K, T, 5]
                gt_trajs=gt_xy,                          # [B, T, 2]
                gt_valid_mask=gt_mask,                   # [B, T]
                pre_nearest_mode_idxs=hard_idx,          # hard assignment fixed by goal candidate
                timestamp_loss_weight=None,
                use_square_gmm=False,
            )

            loss_cls = F.cross_entropy(pred_scores, hard_idx, reduction='none')  # [B]

            layer_loss = (w_reg * loss_reg_gmm + w_cls * loss_cls)               # [B]
            layer_loss = (layer_loss * valid_final).sum() / valid_final.sum().clamp_min(1.0)
            total = total + layer_loss

        return total / len(preds)
    
    # def get_goal_prob_loss(
    #     self,
    #     goal_prob,                  # [B, N] (prior over goal anchors)
    #     goal_anchor,                # [B, N, 2] (anchor positions in target coord)
    #     gt,                         # [B, T, 3] (x, y, valid)
    #     center_gt_final_valid_idx,  # [B]
    # ):
    #     eps = 1e-9
    #     entropy_weight = self.config.get('entropy_weight', 0.3)
    #     kl_weight = self.config.get('kl_weight', 1.0)
    #     sigma = self.config.get('goal_prob_sigma', 2.0)

    #     B, N = goal_prob.shape
    #     device = goal_prob.device
    #     b_idx = torch.arange(B, device=device)

    #     # per-sample final valid GT goal
    #     gt_goal = gt[b_idx, center_gt_final_valid_idx.long(), :2]                  # [B, 2]
    #     valid_final = gt[b_idx, center_gt_final_valid_idx.long(), -1].float()      # [B]

    #     # likelihood p(goal_gt | anchor_n): isotropic Gaussian (in log-space, const dropped)
    #     sq_dist = ((goal_anchor - gt_goal.unsqueeze(1)) ** 2).sum(dim=-1)          # [B, N]
    #     log_lik = -0.5 * sq_dist / (sigma ** 2)                                     # [B, N]

    #     # posterior q(n) ∝ p(goal_gt | n) * p(n)
    #     prior = goal_prob.clamp_min(eps)
    #     log_post_unnorm = log_lik + torch.log(prior)
    #     log_post = log_post_unnorm - torch.logsumexp(log_post_unnorm, dim=-1, keepdim=True)
    #     post_pr = torch.exp(log_post)                                               # [B, N]

    #     # expected negative log-likelihood under posterior
    #     nll = ((-log_lik) * post_pr).sum(dim=-1)                                    # [B]
    #     nll = (nll * valid_final).sum() / valid_final.sum().clamp_min(1.0)

    #     # posterior entropy (encourage confident assignment)
    #     post_entropy = -(post_pr * torch.log(post_pr.clamp_min(eps))).sum(dim=-1)   # [B]
    #     post_entropy = (post_entropy * valid_final).sum() / valid_final.sum().clamp_min(1.0)

    #     # KL(post || prior), same style as nll_loss_multimodes
    #     kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
    #     kl_loss = kl_loss_fn(torch.log(prior), post_pr)

    #     return nll + entropy_weight * post_entropy + kl_weight * kl_loss

    
    # def get_goal_prediction_loss(self, goal_reg, goal_FDE, gt, traj_data):
    #     mask = gt[..., -1]
        
    #     reg_loss = (torch.norm((goal_reg[:, :, :2].permute(1, 0, 2) - gt[:, -1, :2].unsqueeze(1)), 2, dim=-1) * mask[:, -1:])
        
    #     goal_reg_detached = goal_reg.detach()
    #     FDE_gt = (torch.norm(goal_reg_detached[:, :, :2].permute(1, 0, 2) - gt[:, -1, :2].unsqueeze(1), 2, dim=-1) * mask[:, -1:])
    #     disp_loss = self.goal_FDE_loss(goal_FDE, FDE_gt)
        
    #     reg_loss = reg_loss.min(dim=1)[0].mean()

    #     total_loss = self.config['goal_reg_weight'] * reg_loss + self.config['disp_weight'] * disp_loss
    #     return total_loss

    def get_goal_FDE_loss(self, anchor_pos, goal_FDE, gt_decoder, center_gt_final_valid_idx):
        """
        Args:
            anchor_pos: [B, N, 2] goal anchors in target-centric coordinates
            goal_FDE: [B, N] predicted FDE for each anchor
            gt_decoder: [B, T, 3] (x, y, valid)
            center_gt_final_valid_idx: [B] final valid timestep index
        """
        device = gt_decoder.device
        B = gt_decoder.size(0)
        b_idx = torch.arange(B, device=device)

        final_idx = center_gt_final_valid_idx.long()
        gt_goal = gt_decoder[b_idx, final_idx, :2]               # [B, 2]
        valid_final = gt_decoder[b_idx, final_idx, -1].float()   # [B]

        # GT FDE for every anchor: distance(anchor_n, gt_goal_final)
        FDE_gt = (anchor_pos - gt_goal[:, None, :]).norm(dim=-1)  # [B, N]

        valid_mask = valid_final > 0
        if valid_mask.any():
            disp_loss = self.goal_FDE_loss(goal_FDE[valid_mask], FDE_gt[valid_mask])
        else:
            # Keep graph/device/dtype consistent when no valid sample exists.
            disp_loss = goal_FDE.sum() * 0.0

        total_loss = self.config.get('disp_weight', 1.0) * disp_loss
        return total_loss

    def get_dense_future_prediction_loss(self, prediction, gt):
        obj_trajs_future_state = gt['obj_trajs_future_state']
        obj_trajs_future_mask = gt['obj_trajs_future_mask']
        pred_dense_trajs = prediction # (num_center_objects, num_objects, num_future_frames, 7)
        assert pred_dense_trajs.shape[-1] == 7
        assert obj_trajs_future_state.shape[-1] == 4

        pred_dense_trajs_gmm, pred_dense_trajs_vel = pred_dense_trajs[:, :, :, 0:5], pred_dense_trajs[:, :, :, 5:7]

        loss_reg_vel = F.l1_loss(pred_dense_trajs_vel, obj_trajs_future_state[:, :, :, 2:4], reduction='none')
        loss_reg_vel = (loss_reg_vel * obj_trajs_future_mask[:, :, :, None]).sum(dim=-1).sum(dim=-1)

        num_center_objects, num_objects, num_timestamps, _ = pred_dense_trajs.shape
        fake_scores = pred_dense_trajs.new_zeros((num_center_objects, num_objects)).view(-1,
                                                                                         1)  # (num_center_objects * num_objects, 1)

        temp_pred_trajs = pred_dense_trajs_gmm.contiguous().view(num_center_objects * num_objects, 1, num_timestamps, 5)
        temp_gt_idx = torch.zeros(num_center_objects * num_objects).long()  # (num_center_objects * num_objects)
        temp_gt_trajs = obj_trajs_future_state[:, :, :, 0:2].contiguous().view(num_center_objects * num_objects,
                                                                               num_timestamps, 2)
        temp_gt_trajs_mask = obj_trajs_future_mask.view(num_center_objects * num_objects, num_timestamps)
        loss_reg_gmm, _ = self.nll_loss_gmm_direct(
            pred_scores=fake_scores, pred_trajs=temp_pred_trajs, gt_trajs=temp_gt_trajs,
            gt_valid_mask=temp_gt_trajs_mask,
            pre_nearest_mode_idxs=temp_gt_idx,
            timestamp_loss_weight=None, use_square_gmm=False,
        )
        loss_reg_gmm = loss_reg_gmm.view(num_center_objects, num_objects)

        loss_reg = loss_reg_vel + loss_reg_gmm

        obj_valid_mask = obj_trajs_future_mask.sum(dim=-1) > 0

        loss_reg = (loss_reg * obj_valid_mask.float()).sum(dim=-1) / torch.clamp_min(obj_valid_mask.sum(dim=-1),
                                                                                     min=1.0)
        loss_reg = loss_reg.mean()

      
        # return loss_reg * 10.0 # kong_fixme
        # return loss_reg * 3.0
        return loss_reg
    
    def nll_loss_gmm_direct(self, pred_scores, pred_trajs, gt_trajs, gt_valid_mask, pre_nearest_mode_idxs=None,
                            timestamp_loss_weight=None, use_square_gmm=False, log_std_range=(-1.609, 5.0), rho_limit=0.5):
        """
        GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
        Written by Shaoshuai Shi 

        Args:
            pred_scores (batch_size, num_modes):
            pred_trajs (batch_size, num_modes, num_timestamps, 5 or 3)
            gt_trajs (batch_size, num_timestamps, 2):
            gt_valid_mask (batch_size, num_timestamps):
            timestamp_loss_weight (num_timestamps):
        """
        if use_square_gmm:
            assert pred_trajs.shape[-1] == 3
        else:
            assert pred_trajs.shape[-1] == 5

        batch_size = pred_scores.shape[0]

        if pre_nearest_mode_idxs is not None:
            nearest_mode_idxs = pre_nearest_mode_idxs
        else:
            distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :]).norm(dim=-1)
            distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1)

            nearest_mode_idxs = distance.argmin(dim=-1)
        nearest_mode_bs_idxs = torch.arange(batch_size).type_as(nearest_mode_idxs)  # (batch_size, 2)

        nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]  # (batch_size, num_timestamps, 5)
        res_trajs = gt_trajs - nearest_trajs[:, :, 0:2]  # (batch_size, num_timestamps, 2)
        dx = res_trajs[:, :, 0]
        dy = res_trajs[:, :, 1]

        if use_square_gmm:
            log_std1 = log_std2 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
            std1 = std2 = torch.exp(log_std1)  # (0.2m to 150m)
            rho = torch.zeros_like(log_std1)
        else:
            log_std1 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
            log_std2 = torch.clip(nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
            std1 = torch.exp(log_std1)  # (0.2m to 150m)
            std2 = torch.exp(log_std2)  # (0.2m to 150m)
            rho = torch.clip(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

        gt_valid_mask = gt_valid_mask.type_as(pred_scores)
        if timestamp_loss_weight is not None:
            gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

        # -log(a^-1 * e^b) = log(a) - b
        reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho ** 2)  # (batch_size, num_timestamps)
        reg_gmm_exp = (0.5 * 1 / (1 - rho ** 2)) * (
                (dx ** 2) / (std1 ** 2) + (dy ** 2) / (std2 ** 2) - 2 * rho * dx * dy / (
                std1 * std2))  # (batch_size, num_timestamps)

        reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)

        return reg_loss, nearest_mode_idxs
