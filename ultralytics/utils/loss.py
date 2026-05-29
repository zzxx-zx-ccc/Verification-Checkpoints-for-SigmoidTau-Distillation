# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import csv

import numpy as np

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from .metrics import bbox_iou, probiou
from .tal import bbox2dist

# 引入蒸馏相关的超参数
from Globals import (
    bool_distill,
    flag_train_output,
    hyp_T,
    hyp_box_distill,
    hyp_cls_distill,
    hyp_dfl_distill,
    hyp_w_t_cls,
    hyp_w_t_box,
    hyp_w_t_dfl,
    hyp_kd,
    hyp_cls_kd_mode,
    hyp_cls_kd_t2,
    hyp_use_partition_kd,
)
import Globals as G
import sys

class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Compatibility wrapper for checkpoints saved with older Ultralytics loss objects."""

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        return BboxLoss._df_loss(pred_dist, target)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)  # 返回参数用于后续蒸馏
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
            target_ltrb = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl, target_ltrb

    @staticmethod
    def _df_loss(pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max, use_dfl)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        # 蒸馏参数
        self.batch_size = 16
        # Keep temperature on the same device as model parameters to avoid CPU/GPU mismatch.
        self.T = torch.tensor(float(hyp_T), device=device)
        self.w_t_cls_1 = hyp_w_t_cls
        self.w_t_cls_2 = 1.0 - hyp_w_t_cls
        self.w_t_box_1 = hyp_w_t_box
        self.w_t_box_2 = 1.0 - hyp_w_t_box
        self.w_t_dfl_1 = hyp_w_t_dfl
        self.w_t_dfl_2 = 1.0 - hyp_w_t_dfl
        self.w_box_distill = hyp_box_distill
        self.w_cls_distill = hyp_cls_distill
        self.w_dfl_distill = hyp_dfl_distill
        self.kd_weight = hyp_kd
        self.cls_kd_mode = hyp_cls_kd_mode
        self.cls_kd_t2 = hyp_cls_kd_t2
        self.use_partition_kd = hyp_use_partition_kd
        # Optional debug/safety knobs, read with fallback for compatibility.
        self.teacher_conf_gate = float(getattr(G, "hyp_teacher_conf_gate", 0.0))
        self.kd_log_interval = int(getattr(G, "hyp_kd_log_interval", 50))
        self._kd_log_counter = 0

        # KD analysis: train-time sampling + epoch-end plotting.
        self.kd_analysis_enable = bool(getattr(G, "hyp_kd_analysis_enable", False))
        self.kd_analysis_plot_interval = int(getattr(G, "hyp_kd_analysis_plot_interval", 5))
        self.kd_analysis_sample_interval = int(getattr(G, "hyp_kd_analysis_sample_interval", 10))
        self.kd_analysis_conf_bins = int(getattr(G, "hyp_kd_analysis_conf_bins", 50))
        self.kd_analysis_taus = tuple(float(x) for x in getattr(G, "hyp_kd_analysis_taus", [float(hyp_T)]))
        self._kd_analysis_step = 0
        self._kd_analysis_history = []
        self._reset_kd_analysis_epoch()

    def _reset_kd_analysis_epoch(self):
        self._kd_epoch_stats = {
            "samples": 0,
            "region": {"Rt": 0, "Rs": 0, "Ro": 0},
            "area": {
                "small": {"Rt": 0, "Rs": 0, "Ro": 0},
                "medium": {"Rt": 0, "Rs": 0, "Ro": 0},
                "large": {"Rt": 0, "Rs": 0, "Ro": 0},
            },
            "occ": {
                "low": {"Rt": 0, "Rs": 0, "Ro": 0},
                "mid": {"Rt": 0, "Rs": 0, "Ro": 0},
                "high": {"Rt": 0, "Rs": 0, "Ro": 0},
            },
            "hist": {
                "teacher": {
                    "Rt": np.zeros(self.kd_analysis_conf_bins, dtype=np.int64),
                    "Rs": np.zeros(self.kd_analysis_conf_bins, dtype=np.int64),
                    "Ro": np.zeros(self.kd_analysis_conf_bins, dtype=np.int64),
                    "FG": np.zeros(self.kd_analysis_conf_bins, dtype=np.int64),
                    "BG": np.zeros(self.kd_analysis_conf_bins, dtype=np.int64),
                },
                "student": {
                    "Rt": np.zeros(self.kd_analysis_conf_bins, dtype=np.int64),
                    "Rs": np.zeros(self.kd_analysis_conf_bins, dtype=np.int64),
                    "Ro": np.zeros(self.kd_analysis_conf_bins, dtype=np.int64),
                    "FG": np.zeros(self.kd_analysis_conf_bins, dtype=np.int64),
                    "BG": np.zeros(self.kd_analysis_conf_bins, dtype=np.int64),
                },
            },
            "tau_hist": {
                tau: {
                    "teacher": np.zeros(self.kd_analysis_conf_bins, dtype=np.int64),
                    "student": np.zeros(self.kd_analysis_conf_bins, dtype=np.int64),
                }
                for tau in self.kd_analysis_taus
            },
            "conf_hist": np.zeros(self.kd_analysis_conf_bins, dtype=np.int64),
            "gap_hist": np.zeros(self.kd_analysis_conf_bins, dtype=np.int64),
            "step_rows": [],
            "tau_rows": [],
        }

    def _update_hist_counts(self, conf_tensor, mask, out_hist):
        if not mask.any():
            return
        v = conf_tensor[mask].detach().float().clamp_(0.0, 1.0).cpu().numpy()
        if v.size == 0:
            return
        hist, _ = np.histogram(v, bins=self.kd_analysis_conf_bins, range=(0.0, 1.0))
        out_hist += hist.astype(np.int64)

    def _hist_counts_1d(self, values):
        v = values.detach().float().clamp(0.0, 1.0).cpu().numpy()
        hist, _ = np.histogram(v, bins=self.kd_analysis_conf_bins, range=(0.0, 1.0))
        return hist.astype(np.int64)

    def _safe_quantiles(self, values):
        if values.numel() == 0:
            return 0.0, 0.0, 0.0
        q = torch.quantile(values.detach().float(), torch.tensor([0.1, 0.5, 0.9], device=values.device))
        return tuple(float(x) for x in q)

    def _box_iou_matrix(self, boxes1, boxes2):
        lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = (rb - lt).clamp_min(0)
        inter = wh[..., 0] * wh[..., 1]
        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)
        union = area1[:, None] + area2[None, :] - inter
        return inter / union.clamp_min(1e-9)

    def _compute_anchor_occ_score(self, boxes_xyxy, fg_union):
        # Approximate occlusion by max IoU with other unique GT boxes in the same image.
        occ = torch.zeros(boxes_xyxy.shape[:2], device=boxes_xyxy.device, dtype=boxes_xyxy.dtype)
        bsz = boxes_xyxy.shape[0]
        for bi in range(bsz):
            sel = fg_union[bi]
            if not sel.any():
                continue
            boxes = boxes_xyxy[bi][sel]
            if boxes.shape[0] < 2:
                continue
            uniq, inv = torch.unique(boxes, dim=0, return_inverse=True)
            if uniq.shape[0] < 2:
                continue
            iou = self._box_iou_matrix(uniq, uniq)
            iou.fill_diagonal_(0)
            max_iou = iou.max(dim=1).values
            occ[bi][sel] = max_iou[inv]
        return occ

    def _collect_kd_analysis(
        self,
        rt,
        rs,
        ro,
        fg_teacher,
        fg_student,
        target_scores_teacher,
        target_scores_student,
        target_bboxes_student,
        pred_scores_teacher,
        pred_scores_student,
    ):
        if not self.kd_analysis_enable:
            return
        self._kd_analysis_step += 1
        if self.kd_analysis_sample_interval > 1 and (self._kd_analysis_step % self.kd_analysis_sample_interval != 0):
            return

        fg_union = fg_teacher | fg_student
        if not fg_union.any():
            return

        st = self._kd_epoch_stats
        st["samples"] += 1
        st["region"]["Rt"] += int(rt.sum().item())
        st["region"]["Rs"] += int(rs.sum().item())
        st["region"]["Ro"] += int(ro.sum().item())
        n_rt = int(rt.sum().item())
        n_rs = int(rs.sum().item())
        n_ro = int(ro.sum().item())
        w_rt = float(target_scores_teacher[rt].sum().item()) if rt.any() else 0.0
        w_rs = float(target_scores_student[rs].sum().item()) if rs.any() else 0.0
        w_ro = float(target_scores_teacher[ro].sum().item()) if ro.any() else 0.0
        w_total = max(w_rt + w_rs + w_ro, 1e-12)

        wh = (target_bboxes_student[..., 2:4] - target_bboxes_student[..., 0:2]).clamp_min(0)
        area = wh[..., 0] * wh[..., 1]
        small = area < (32.0 ** 2)
        medium = (area >= (32.0 ** 2)) & (area < (96.0 ** 2))
        large = area >= (96.0 ** 2)
        for name, m in (("small", small), ("medium", medium), ("large", large)):
            st["area"][name]["Rt"] += int((rt & m).sum().item())
            st["area"][name]["Rs"] += int((rs & m).sum().item())
            st["area"][name]["Ro"] += int((ro & m).sum().item())

        occ = self._compute_anchor_occ_score(target_bboxes_student, fg_union)
        occ_low = occ < 0.2
        occ_mid = (occ >= 0.2) & (occ < 0.5)
        occ_high = occ >= 0.5
        for name, m in (("low", occ_low), ("mid", occ_mid), ("high", occ_high)):
            st["occ"][name]["Rt"] += int((rt & m).sum().item())
            st["occ"][name]["Rs"] += int((rs & m).sum().item())
            st["occ"][name]["Ro"] += int((ro & m).sum().item())

        tconf = torch.sigmoid(pred_scores_teacher).max(-1).values
        sconf = torch.sigmoid(pred_scores_student).max(-1).values
        self._update_hist_counts(tconf, rt, st["hist"]["teacher"]["Rt"])
        self._update_hist_counts(tconf, rs, st["hist"]["teacher"]["Rs"])
        self._update_hist_counts(tconf, ro, st["hist"]["teacher"]["Ro"])
        self._update_hist_counts(sconf, rt, st["hist"]["student"]["Rt"])
        self._update_hist_counts(sconf, rs, st["hist"]["student"]["Rs"])
        self._update_hist_counts(sconf, ro, st["hist"]["student"]["Ro"])
        self._update_hist_counts(tconf, fg_union, st["hist"]["teacher"]["FG"])
        self._update_hist_counts(tconf, ~fg_union, st["hist"]["teacher"]["BG"])
        self._update_hist_counts(sconf, fg_union, st["hist"]["student"]["FG"])
        self._update_hist_counts(sconf, ~fg_union, st["hist"]["student"]["BG"])

        r_teacher = rt | ro
        conf = torch.empty(0, device=pred_scores_teacher.device)
        gap = torch.empty(0, device=pred_scores_teacher.device)
        if r_teacher.any():
            t_logits = pred_scores_teacher[r_teacher].detach()
            s_logits = pred_scores_student[r_teacher].detach()
            pt = torch.sigmoid(t_logits)
            ps = torch.sigmoid(s_logits)
            conf = pt.max(-1).values
            gap = (pt - ps).abs().max(-1).values
            st["conf_hist"] += self._hist_counts_1d(conf)
            st["gap_hist"] += self._hist_counts_1d(gap)

            bin_edges = np.linspace(0.0, 1.0, self.kd_analysis_conf_bins + 1)
            for tau in self.kd_analysis_taus:
                tau_tensor = torch.tensor(tau, device=t_logits.device, dtype=t_logits.dtype).clamp_min(1e-6)
                t_resp = torch.sigmoid(t_logits - torch.log(tau_tensor)).flatten()
                s_resp = torch.sigmoid(s_logits - torch.log(tau_tensor)).flatten()
                hist_t = self._hist_counts_1d(t_resp)
                hist_s = self._hist_counts_1d(s_resp)
                st["tau_hist"][tau]["teacher"] += hist_t
                st["tau_hist"][tau]["student"] += hist_s
                for bi, (left, right) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                    st["tau_rows"].append(
                        {
                            "step": self._kd_analysis_step,
                            "tau": tau,
                            "bin": bi,
                            "bin_left": float(left),
                            "bin_right": float(right),
                            "teacher_count": int(hist_t[bi]),
                            "student_count": int(hist_s[bi]),
                        }
                    )

        conf_q10, conf_q50, conf_q90 = self._safe_quantiles(conf)
        gap_q10, gap_q50, gap_q90 = self._safe_quantiles(gap)
        st["step_rows"].append(
            {
                "step": self._kd_analysis_step,
                "rt": n_rt,
                "rs": n_rs,
                "ro": n_ro,
                "rt_ratio": n_rt / max(n_rt + n_rs + n_ro, 1),
                "rs_ratio": n_rs / max(n_rt + n_rs + n_ro, 1),
                "ro_ratio": n_ro / max(n_rt + n_rs + n_ro, 1),
                "w_rt": w_rt,
                "w_rs": w_rs,
                "w_ro": w_ro,
                "w_rt_ratio": w_rt / w_total,
                "w_rs_ratio": w_rs / w_total,
                "w_ro_ratio": w_ro / w_total,
                "teacher_conf_mean": float(conf.mean().item()) if conf.numel() else 0.0,
                "teacher_conf_q10": conf_q10,
                "teacher_conf_q50": conf_q50,
                "teacher_conf_q90": conf_q90,
                "gap_mean": float(gap.mean().item()) if gap.numel() else 0.0,
                "gap_q10": gap_q10,
                "gap_q50": gap_q50,
                "gap_q90": gap_q90,
            }
        )

    def _save_stacked_bar(self, dist_dict, x_labels, out_path, title):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rt = [dist_dict[x]["Rt"] for x in x_labels]
        rs = [dist_dict[x]["Rs"] for x in x_labels]
        ro = [dist_dict[x]["Ro"] for x in x_labels]

        x = np.arange(len(x_labels))
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
        ax.bar(x, rt, label="Rt", color="#e76f51")
        ax.bar(x, rs, bottom=rt, label="Rs", color="#2a9d8f")
        ax.bar(x, ro, bottom=(np.array(rt) + np.array(rs)), label="Ro", color="#457b9d")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    def _save_region_conf_hist(self, hist_pack, out_path, title):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bins = np.linspace(0.0, 1.0, self.kd_analysis_conf_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.8), dpi=140, sharey=True)
        for idx, region in enumerate(("Rt", "Rs", "Ro")):
            t = hist_pack["teacher"][region].astype(np.float64)
            s = hist_pack["student"][region].astype(np.float64)
            t = t / max(t.sum(), 1.0)
            s = s / max(s.sum(), 1.0)
            axs[idx].plot(centers, t, label="Teacher", color="#f4a261", linewidth=1.6)
            axs[idx].plot(centers, s, label="Student", color="#264653", linewidth=1.6)
            axs[idx].set_title(region)
            axs[idx].set_xlabel("Confidence")
            axs[idx].grid(alpha=0.2)
        axs[0].set_ylabel("Density")
        axs[-1].legend(frameon=False)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    def _save_fg_bg_conf_hist(self, hist_pack, out_path, title):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bins = np.linspace(0.0, 1.0, self.kd_analysis_conf_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.8), dpi=140, sharey=True)
        for idx, who in enumerate(("teacher", "student")):
            fg = hist_pack[who]["FG"].astype(np.float64)
            bg = hist_pack[who]["BG"].astype(np.float64)
            fg = fg / max(fg.sum(), 1.0)
            bg = bg / max(bg.sum(), 1.0)
            axs[idx].plot(centers, fg, label="FG", color="#e63946", linewidth=1.6)
            axs[idx].plot(centers, bg, label="BG", color="#1d3557", linewidth=1.6)
            axs[idx].set_title("Teacher" if who == "teacher" else "Student")
            axs[idx].set_xlabel("Confidence")
            axs[idx].grid(alpha=0.2)
            axs[idx].legend(frameon=False)
        axs[0].set_ylabel("Density")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    def _save_tau_response_hist(self, tau_hist, out_path, title):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bins = np.linspace(0.0, 1.0, self.kd_analysis_conf_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        n = max(len(tau_hist), 1)
        fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 3.8), dpi=140, sharey=True)
        axs = np.atleast_1d(axs)
        for ax, tau in zip(axs, sorted(tau_hist)):
            t = tau_hist[tau]["teacher"].astype(np.float64)
            s = tau_hist[tau]["student"].astype(np.float64)
            t = t / max(t.sum(), 1.0)
            s = s / max(s.sum(), 1.0)
            ax.plot(centers, t, label="Teacher", color="#f4a261", linewidth=1.6)
            ax.plot(centers, s, label="Student", color="#264653", linewidth=1.6)
            ax.set_title(f"tau={tau:g}")
            ax.set_xlabel("sigmoid(logit - log(tau))")
            ax.grid(alpha=0.2)
        axs[0].set_ylabel("Density")
        axs[-1].legend(frameon=False)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    def _save_conf_gap_hist(self, conf_hist, gap_hist, out_path, title):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bins = np.linspace(0.0, 1.0, self.kd_analysis_conf_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.8), dpi=140, sharey=True)
        for ax, hist, name, color in (
            (axs[0], conf_hist, "Teacher confidence", "#e76f51"),
            (axs[1], gap_hist, "Student-teacher gap", "#2a9d8f"),
        ):
            v = hist.astype(np.float64)
            v = v / max(v.sum(), 1.0)
            ax.plot(centers, v, color=color, linewidth=1.8)
            ax.set_title(name)
            ax.set_xlabel("Value")
            ax.grid(alpha=0.2)
        axs[0].set_ylabel("Density")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    def _append_csv_rows(self, path, rows, fieldnames, epoch_id):
        if not rows:
            return
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                out = {"epoch": epoch_id}
                out.update(row)
                writer.writerow(out)

    def _save_region_ratio_trend(self, out_path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self._kd_analysis_history:
            return
        epochs = [x["epoch"] for x in self._kd_analysis_history]
        fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=140)
        for region, color in (("Rt", "#e76f51"), ("Rs", "#2a9d8f"), ("Ro", "#457b9d")):
            ratio = [x["region_ratio"][region] for x in self._kd_analysis_history]
            ax.plot(epochs, ratio, marker="o", linewidth=1.6, markersize=3.5, label=region, color=color)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Ratio")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Rt/Rs/Ro Ratio Trend")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    def flush_kd_analysis_epoch(self, save_dir, epoch, force=False):
        if not self.kd_analysis_enable:
            return
        st = self._kd_epoch_stats
        if st["samples"] <= 0:
            return

        save_root = Path(save_dir) / "kd_analysis"
        save_root.mkdir(parents=True, exist_ok=True)
        epoch_id = int(epoch) + 1

        total_region = st["region"]["Rt"] + st["region"]["Rs"] + st["region"]["Ro"]
        region_ratio = {
            "Rt": st["region"]["Rt"] / max(total_region, 1),
            "Rs": st["region"]["Rs"] / max(total_region, 1),
            "Ro": st["region"]["Ro"] / max(total_region, 1),
        }
        self._kd_analysis_history.append({"epoch": epoch_id, "region": dict(st["region"]), "region_ratio": region_ratio})

        # Save epoch summary as csv-friendly line.
        summary_path = save_root / "kd_epoch_summary.csv"
        write_header = not summary_path.exists()
        with open(summary_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write(
                    "epoch,samples,rt,rs,ro,rt_ratio,rs_ratio,ro_ratio,"
                    "rt_small,rt_medium,rt_large,rs_small,rs_medium,rs_large,ro_small,ro_medium,ro_large,"
                    "rt_occ_low,rt_occ_mid,rt_occ_high,rs_occ_low,rs_occ_mid,rs_occ_high,ro_occ_low,ro_occ_mid,ro_occ_high\n"
                )
            f.write(
                f"{epoch_id},{st['samples']},{st['region']['Rt']},{st['region']['Rs']},{st['region']['Ro']},"
                f"{region_ratio['Rt']:.6f},{region_ratio['Rs']:.6f},{region_ratio['Ro']:.6f},"
                f"{st['area']['small']['Rt']},{st['area']['medium']['Rt']},{st['area']['large']['Rt']},"
                f"{st['area']['small']['Rs']},{st['area']['medium']['Rs']},{st['area']['large']['Rs']},"
                f"{st['area']['small']['Ro']},{st['area']['medium']['Ro']},{st['area']['large']['Ro']},"
                f"{st['occ']['low']['Rt']},{st['occ']['mid']['Rt']},{st['occ']['high']['Rt']},"
                f"{st['occ']['low']['Rs']},{st['occ']['mid']['Rs']},{st['occ']['high']['Rs']},"
                f"{st['occ']['low']['Ro']},{st['occ']['mid']['Ro']},{st['occ']['high']['Ro']}\n"
            )

        self._append_csv_rows(
            save_root / "kd_step_stats.csv",
            st["step_rows"],
            [
                "epoch",
                "step",
                "rt",
                "rs",
                "ro",
                "rt_ratio",
                "rs_ratio",
                "ro_ratio",
                "w_rt",
                "w_rs",
                "w_ro",
                "w_rt_ratio",
                "w_rs_ratio",
                "w_ro_ratio",
                "teacher_conf_mean",
                "teacher_conf_q10",
                "teacher_conf_q50",
                "teacher_conf_q90",
                "gap_mean",
                "gap_q10",
                "gap_q50",
                "gap_q90",
            ],
            epoch_id,
        )
        self._append_csv_rows(
            save_root / "kd_tau_hist.csv",
            st["tau_rows"],
            ["epoch", "step", "tau", "bin", "bin_left", "bin_right", "teacher_count", "student_count"],
            epoch_id,
        )

        do_plot = force or self.kd_analysis_plot_interval <= 1 or (epoch_id % self.kd_analysis_plot_interval == 0)
        if do_plot:
            self._save_stacked_bar(
                st["area"],
                ["small", "medium", "large"],
                save_root / f"epoch_{epoch_id:04d}_area_rt_rs_ro.png",
                f"Area Buckets (Epoch {epoch_id})",
            )
            self._save_stacked_bar(
                st["occ"],
                ["low", "mid", "high"],
                save_root / f"epoch_{epoch_id:04d}_occlusion_rt_rs_ro.png",
                f"Occlusion Buckets (Epoch {epoch_id})",
            )
            self._save_region_conf_hist(
                st["hist"],
                save_root / f"epoch_{epoch_id:04d}_region_conf_hist.png",
                f"Teacher vs Student by Region (Epoch {epoch_id})",
            )
            self._save_fg_bg_conf_hist(
                st["hist"],
                save_root / f"epoch_{epoch_id:04d}_fg_bg_conf_hist.png",
                f"Foreground vs Background Confidence (Epoch {epoch_id})",
            )
            self._save_tau_response_hist(
                st["tau_hist"],
                save_root / f"epoch_{epoch_id:04d}_tau_response_hist.png",
                f"Teacher vs Student Response under tau (Epoch {epoch_id})",
            )
            self._save_conf_gap_hist(
                st["conf_hist"],
                st["gap_hist"],
                save_root / f"epoch_{epoch_id:04d}_conf_gap_hist.png",
                f"Teacher Confidence and Student-Teacher Gap (Epoch {epoch_id})",
            )
            self._save_region_ratio_trend(save_root / "region_ratio_trend.png")

        LOGGER.info(
            f"KD Analysis | epoch={epoch_id} samples={st['samples']} "
            f"Rt/Rs/Ro={st['region']['Rt']}/{st['region']['Rs']}/{st['region']['Ro']} "
            f"plots={'on' if do_plot else 'off'}"
        )
        self._reset_kd_analysis_epoch()

    def on_train_epoch_end(self, epoch, save_dir=None, force=False):
        if save_dir is None:
            save_dir = Path("runs/detect") / "kd_analysis_fallback"
        self.flush_kd_analysis_epoch(save_dir, epoch, force=force)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = (pred_dist.view(b, a, 4, c // 4)).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    # 原版loss计算
    def loss_calculate(self, feats, batch):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        C = self.bce(pred_scores, target_scores.to(dtype)).sum(-1) / target_scores_sum  # BCE
        loss[1] = C.sum()

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2], target_ltrb = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            target_ltrb = torch.tensor(0.0).to(self.device)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        # loss(box, cls, dfl)
        return loss.sum() * batch_size, loss.detach(), [pred_scores, fg_mask, target_scores, batch_size, pred_distri, pred_bboxes, target_bboxes, target_ltrb]

# 改进：
    # 蒸馏loss计算
    def D_loss_calculate(self, distill_relative_teacher, distill_relative_student):

        # [1]获取蒸馏相关参数
        [pred_scores_teacher, fg_mask_teacher, target_scores_teacher, batch_size_teacher,
         pred_distri_teacher, pred_bboxes_teacher, target_bboxes_teacher, target_ltrb_teacher] = distill_relative_teacher
        [pred_scores_student, fg_mask_student, target_scores_student, batch_size_studnet,
         pred_distri_student, pred_bboxes_student, target_bboxes_student, target_ltrb_student] = distill_relative_student

        # [2]数据准备
        fg_mask_cls = fg_mask_teacher
        fg_mask_loc = fg_mask_teacher & fg_mask_student

        if not fg_mask_cls.any() and not fg_mask_loc.any():
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero, zero
        self.batch_size = batch_size_teacher if batch_size_teacher == batch_size_studnet else batch_size_studnet  # 公共的batch大小

        # Rt/Rs/Ro partition used by analysis and (optionally) partition KD.
        rt = fg_mask_teacher & (~fg_mask_student)   # teacher-only
        rs = fg_mask_student & (~fg_mask_teacher)   # student-only
        ro = fg_mask_teacher & fg_mask_student      # overlap
        self._collect_kd_analysis(
            rt=rt,
            rs=rs,
            ro=ro,
            fg_teacher=fg_mask_teacher,
            fg_student=fg_mask_student,
            target_scores_teacher=target_scores_teacher,
            target_scores_student=target_scores_student,
            target_bboxes_student=target_bboxes_student,
            pred_scores_teacher=pred_scores_teacher,
            pred_scores_student=pred_scores_student,
        )

        # Legacy path (default): keep previous behavior for backward compatibility.
        if not self.use_partition_kd:
            # [3]蒸馏cls损失
            if fg_mask_cls.any():
                temp = self._temperature_like(pred_scores_student)
                student_logits_cls = pred_scores_student[fg_mask_cls]
                teacher_logits_cls = pred_scores_teacher[fg_mask_cls].detach()

                # Optional confidence gate on teacher predictions for cleaner KD targets.
                keep = None
                if self.teacher_conf_gate > 0:
                    tconf = torch.sigmoid(teacher_logits_cls).max(-1).values
                    keep = tconf >= self.teacher_conf_gate
                    if keep.any():
                        student_logits_cls = student_logits_cls[keep]
                        teacher_logits_cls = teacher_logits_cls[keep]
                    else:
                        student_logits_cls = student_logits_cls[:0]
                        teacher_logits_cls = teacher_logits_cls[:0]

                if student_logits_cls.shape[0] > 0:
                    loss_cls_base, loss_cls_temp = self._cls_distill_losses(student_logits_cls, teacher_logits_cls, temp)
                    loss_cls_t = self.w_t_cls_1 * loss_cls_temp + self.w_t_cls_2 * loss_cls_base

                    cls_weight = target_scores_teacher[fg_mask_cls].max(-1).values.detach().clamp_min(0.05)
                    if keep is not None:
                        cls_weight = cls_weight[keep]
                    distill_loss_cls = (loss_cls_t * cls_weight).sum() / cls_weight.sum().clamp_min(1.0)
                else:
                    distill_loss_cls = torch.tensor(0.0, device=self.device)
            else:
                distill_loss_cls = torch.tensor(0.0, device=self.device)

            # [4]蒸馏bbox损失
            if fg_mask_loc.any():
                pred_bboxes_student_3 = pred_bboxes_student[fg_mask_loc]
                target_bboxes_student_3_1 = pred_bboxes_teacher[fg_mask_loc].detach()
                iou_3_1 = bbox_iou(pred_bboxes_student_3, target_bboxes_student_3_1, xywh=False, CIoU=True)
                weight_box_3 = target_scores_teacher[fg_mask_loc].sum(-1).unsqueeze(-1).detach().clamp_min(0.05)
                distill_loss_box = ((1.0 - iou_3_1) * weight_box_3).sum() / weight_box_3.sum().clamp_min(1.0)
            else:
                distill_loss_box = torch.tensor(0.0, device=self.device)

            # [5]蒸馏dfl损失
            if fg_mask_loc.any():
                shape = target_ltrb_student[fg_mask_loc].shape
                pred_distri_student_3_1 = pred_distri_student[fg_mask_loc].view(-1, self.reg_max)
                target_dfl_distill_3_1 = F.softmax(
                    pred_distri_teacher[fg_mask_loc].detach().view(-1, self.reg_max), dim=-1
                )
                distill_loss_dfl_3_1 = F.cross_entropy(
                    pred_distri_student_3_1, target_dfl_distill_3_1, reduction="none"
                ).view(shape).mean(-1, keepdim=True)
                weight_dfl_3 = target_scores_teacher[fg_mask_loc].sum(-1).unsqueeze(-1).detach().clamp_min(0.05)
                distill_loss_dfl = (distill_loss_dfl_3_1 * weight_dfl_3).sum() / weight_dfl_3.sum().clamp_min(1.0)
            else:
                distill_loss_dfl = torch.tensor(0.0, device=self.device)

            distill_loss_box *= self.w_box_distill
            distill_loss_cls *= self.w_cls_distill
            distill_loss_dfl *= self.w_dfl_distill
            return distill_loss_cls, distill_loss_box, distill_loss_dfl

        # Partition-KD path (Rt/Rs/Ro), no extra tunable hyperparameters.
        r_teacher = rt | ro                         # where teacher-guided cls distill is applied

        # [3] cls distill on teacher-positive regions only (Rt U Ro)
        if r_teacher.any():
            temp = self._temperature_like(pred_scores_student)
            s_logits = pred_scores_student[r_teacher]
            t_logits = pred_scores_teacher[r_teacher].detach()

            if self.teacher_conf_gate > 0:
                tconf = torch.sigmoid(t_logits).max(-1).values
                keep = tconf >= self.teacher_conf_gate
                if keep.any():
                    s_logits = s_logits[keep]
                    t_logits = t_logits[keep]
                else:
                    s_logits = s_logits[:0]
                    t_logits = t_logits[:0]

            if s_logits.shape[0] > 0:
                loss_cls_base, loss_cls_temp = self._cls_distill_losses(s_logits, t_logits, temp)
                cls_loss_each = self.w_t_cls_1 * loss_cls_temp + self.w_t_cls_2 * loss_cls_base

                # auto weights: normalize(max_score_t * max_gap), no new hyperparameters.
                ps = torch.sigmoid(s_logits)
                pt = torch.sigmoid(t_logits)
                max_score_t = pt.max(-1).values
                max_gap = (pt - ps).abs().max(-1).values
                cls_weight = (max_score_t * max_gap).detach()
                cls_weight = cls_weight / cls_weight.mean().clamp_min(1e-6)
                distill_loss_cls = (cls_loss_each * cls_weight).sum() / cls_weight.sum().clamp_min(1.0)
            else:
                distill_loss_cls = torch.tensor(0.0, device=self.device)
        else:
            distill_loss_cls = torch.tensor(0.0, device=self.device)

        # [4] loc distill with region-wise dual guidance, no q.
        loc_num = torch.tensor(0.0, device=self.device)
        loc_den = torch.tensor(0.0, device=self.device)

        if rt.any():
            iou_t = bbox_iou(pred_bboxes_student[rt], pred_bboxes_teacher[rt].detach(), xywh=False, CIoU=True)
            w_rt = target_scores_teacher[rt].sum(-1).unsqueeze(-1).detach().clamp_min(0.05)
            loc_num += ((1.0 - iou_t) * w_rt).sum()
            loc_den += w_rt.sum()

        if rs.any():
            iou_s = bbox_iou(pred_bboxes_student[rs], target_bboxes_student[rs].detach(), xywh=False, CIoU=True)
            w_rs = target_scores_student[rs].sum(-1).unsqueeze(-1).detach().clamp_min(0.05)
            loc_num += ((1.0 - iou_s) * w_rs).sum()
            loc_den += w_rs.sum()

        if ro.any():
            iou_ot = bbox_iou(pred_bboxes_student[ro], pred_bboxes_teacher[ro].detach(), xywh=False, CIoU=True)
            iou_og = bbox_iou(pred_bboxes_student[ro], target_bboxes_student[ro].detach(), xywh=False, CIoU=True)
            loss_ro = 0.5 * (1.0 - iou_ot) + 0.5 * (1.0 - iou_og)
            w_ro = target_scores_teacher[ro].sum(-1).unsqueeze(-1).detach().clamp_min(0.05)
            loc_num += (loss_ro * w_ro).sum()
            loc_den += w_ro.sum()

        distill_loss_box = loc_num / loc_den.clamp_min(1.0)

        # [5] dfl distill with region-wise targets (teacher / gt / 0.5 mix)
        dfl_num = torch.tensor(0.0, device=self.device)
        dfl_den = torch.tensor(0.0, device=self.device)

        if rt.any():
            shape_rt = target_ltrb_student[rt].shape
            pred_rt = pred_distri_student[rt].view(-1, self.reg_max)
            tgt_rt = F.softmax(pred_distri_teacher[rt].detach().view(-1, self.reg_max), dim=-1)
            dfl_rt = F.cross_entropy(pred_rt, tgt_rt, reduction="none").view(shape_rt).mean(-1, keepdim=True)
            w_rt = target_scores_teacher[rt].sum(-1).unsqueeze(-1).detach().clamp_min(0.05)
            dfl_num += (dfl_rt * w_rt).sum()
            dfl_den += w_rt.sum()

        if rs.any():
            shape_rs = target_ltrb_student[rs].shape
            pred_rs = pred_distri_student[rs].view(-1, self.reg_max)
            tgt_rs = self._make_soft_dfl_target(target_ltrb_student[rs].view(-1, 1), pred_rs.shape[-1])
            dfl_rs = F.cross_entropy(pred_rs, tgt_rs, reduction="none").view(shape_rs).mean(-1, keepdim=True)
            w_rs = target_scores_student[rs].sum(-1).unsqueeze(-1).detach().clamp_min(0.05)
            dfl_num += (dfl_rs * w_rs).sum()
            dfl_den += w_rs.sum()

        if ro.any():
            shape_ro = target_ltrb_student[ro].shape
            pred_ro = pred_distri_student[ro].view(-1, self.reg_max)
            tgt_t = F.softmax(pred_distri_teacher[ro].detach().view(-1, self.reg_max), dim=-1)
            tgt_g = self._make_soft_dfl_target(target_ltrb_student[ro].view(-1, 1), pred_ro.shape[-1])
            tgt_ro = 0.5 * tgt_t + 0.5 * tgt_g
            dfl_ro = F.cross_entropy(pred_ro, tgt_ro, reduction="none").view(shape_ro).mean(-1, keepdim=True)
            w_ro = target_scores_teacher[ro].sum(-1).unsqueeze(-1).detach().clamp_min(0.05)
            dfl_num += (dfl_ro * w_ro).sum()
            dfl_den += w_ro.sum()

        distill_loss_dfl = dfl_num / dfl_den.clamp_min(1.0)

        # [6]参数修正
        distill_loss_box *= self.w_box_distill
        distill_loss_cls *= self.w_cls_distill
        distill_loss_dfl *= self.w_dfl_distill

        return distill_loss_cls, distill_loss_box, distill_loss_dfl

    def sigmoid_T(self, x, T):
        # Equivalent to sigmoid(x - log(T)), implemented stably.
        return torch.sigmoid(x - torch.log(T))

    def _cls_distill_losses(self, student_logits_cls, teacher_logits_cls, temp):
        # Base term is always BCE at T=1 to preserve compatibility with previous behavior.
        teacher_probs_base = torch.sigmoid(teacher_logits_cls).detach().clamp(1e-4, 1 - 1e-4)
        loss_cls_base = self.bce(student_logits_cls, teacher_probs_base).mean(-1)

        mode = str(self.cls_kd_mode).lower()
        t2_scale = temp * temp if self.cls_kd_t2 else 1.0

        if mode in {"legacy_shift", "sigmoid_tau"}:
            teacher_probs_temp = self.sigmoid_T(teacher_logits_cls, temp).detach().clamp(1e-4, 1 - 1e-4)
            student_logits_temp = student_logits_cls - torch.log(temp)
            loss_cls_temp = self.bce(student_logits_temp, teacher_probs_temp).mean(-1)
            return loss_cls_base, loss_cls_temp

        if mode == "bce_divt":
            teacher_probs_temp = torch.sigmoid(teacher_logits_cls / temp).detach().clamp(1e-4, 1 - 1e-4)
            student_logits_temp = student_logits_cls / temp
            loss_cls_temp = self.bce(student_logits_temp, teacher_probs_temp).mean(-1) * t2_scale
            return loss_cls_base, loss_cls_temp

        if mode == "kl_divt":
            eps = 1e-6
            teacher_probs_temp = torch.sigmoid(teacher_logits_cls / temp).detach().clamp(eps, 1 - eps)
            student_probs_temp = torch.sigmoid(student_logits_cls / temp).clamp(eps, 1 - eps)
            kl = (
                teacher_probs_temp * (torch.log(teacher_probs_temp) - torch.log(student_probs_temp))
                + (1 - teacher_probs_temp)
                * (torch.log(1 - teacher_probs_temp) - torch.log(1 - student_probs_temp))
            )
            loss_cls_temp = kl.mean(-1) * t2_scale
            return loss_cls_base, loss_cls_temp

        if mode == "bckd_cls":
            # BCKD-style cls branch: binary BCE on sigmoid probabilities.
            student_probs = torch.sigmoid(student_logits_cls).clamp(1e-6, 1 - 1e-6)
            teacher_probs = torch.sigmoid(teacher_logits_cls).detach().clamp(1e-6, 1 - 1e-6)
            loss_bckd = self.bce_WithoutLogits(student_probs, teacher_probs).mean(-1)
            return loss_bckd, loss_bckd

        if mode == "dual_kl_fusion":
            # Analytical fusion of two teacher views:
            # A) expected-polarity margin shift in logit space
            # B) standard temperature scaling in logit space
            # logit* = 0.5 * ( logit_A + logit_B )
            temp = temp.clamp_min(1.001)
            margin = torch.log(temp)
            teacher_probs = torch.sigmoid(teacher_logits_cls).detach()
            expected_polarity = 2.0 * teacher_probs - 1.0
            logit_bayes_shift = teacher_logits_cls - expected_polarity * margin
            logit_temp = teacher_logits_cls / temp
            logit_optimal = 0.5 * (logit_bayes_shift + logit_temp)
            teacher_probs_optimal = torch.sigmoid(logit_optimal).detach().clamp(1e-4, 1 - 1e-4)
            loss_cls_temp = self.bce(student_logits_cls, teacher_probs_optimal).mean(-1)
            # For this mode we use only the fused target, so make weighted blend invariant to hyp_w_t_cls.
            return loss_cls_temp, loss_cls_temp

        # Fallback to the paper-aligned Sigmoid-τ behavior for unexpected values.
        teacher_probs_temp = self.sigmoid_T(teacher_logits_cls, temp).detach().clamp(1e-4, 1 - 1e-4)
        student_logits_temp = student_logits_cls - torch.log(temp)
        loss_cls_temp = self.bce(student_logits_temp, teacher_probs_temp).mean(-1)
        return loss_cls_base, loss_cls_temp

    def bce_WithoutLogits(self, x, y):
        eps = 1e-6
        x = x.clamp(min=eps, max=1 - eps)
        y = y.clamp(min=eps, max=1 - eps)
        return -(y * torch.log(x) + (1 - y) * torch.log(1 - x))

    def _temperature_like(self, ref_tensor):
        if torch.is_tensor(self.T):
            temp = self.T.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        else:
            temp = torch.tensor(float(self.T), device=ref_tensor.device, dtype=ref_tensor.dtype)
        return temp.clamp_min(1e-6)

    def _make_soft_dfl_target(self, target_ltrb_flat, num_bins):
        """Build soft DFL target distribution from continuous ltrb targets."""
        target = target_ltrb_flat.clamp(0, num_bins - 1 - 1e-3)
        tl = target.floor().long()
        tr = (tl + 1).clamp(max=num_bins - 1)
        wl = tr.float() - target
        wr = 1.0 - wl
        out = torch.zeros((target.shape[0], num_bins), device=target.device, dtype=target.dtype)
        out.scatter_(dim=-1, index=tl, src=wl)
        out.scatter_add_(dim=-1, index=tr, src=wr)
        return out

    def gain_T(self, loss_teacher, loss_student):
        import math
        w = (loss_student[1] - loss_teacher[1]) / (loss_student[1] + loss_student[1] - loss_teacher[1])
        start = 0.43  # 初始状态
        x = (start - w) / start
        T1 = 10
        T0 = 0.1
        T = (T1 - T0) / 2 * torch.sin(2 * math.pi * x) + (T1 + T0) / 2  # 计算对应温度
        return T

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""

        # 非蒸馏模式下计算损失
        if not bool_distill:
            feats = preds[1] if isinstance(preds, tuple) else preds
            loss_sum, loss, _ = self.loss_calculate(feats, batch)
            return loss_sum, loss

        else:
            if isinstance(preds, tuple):
                # val过程
                # feats_teacher = preds[1][0]  # 教师通道
                feats_student = preds[1][1]  # 学生通道
                loss_student_sum, loss_student, _ = self.loss_calculate(feats_student, batch)
                return loss_student_sum, loss_student
            else:
                # 训练过程
                feats_teacher = preds[0]  # 教师通道
                feats_student = preds[1]  # 学生通道

            loss = torch.zeros(3, device=self.device)  # box, cls, dfl

            if flag_train_output == 1:
                loss_sum, loss, _ = self.loss_calculate(feats_teacher, batch)
                return loss_sum, loss
            elif flag_train_output == 2:
                loss_sum, loss, _ = self.loss_calculate(feats_student, batch)
                return loss_sum, loss
            else:
                # [1]获取教师模型的相关蒸馏参数
                with torch.no_grad():
                    _, _, distill_relative_teacher = self.loss_calculate(feats_teacher, batch)

                # [2]获取学生模型的相关蒸馏参数
                loss_student_sum, loss_student, distill_relative_student = self.loss_calculate(feats_student, batch)

                # # [3]计算此时的温度T
                # self.T = self.gain_T(loss_teacher, loss_student)

                # [4]计算蒸馏loss
                distill_loss_cls, distill_loss_box, distill_loss_dfl = self.D_loss_calculate(distill_relative_teacher, distill_relative_student)

                # [5]数据整理
                loss[0] = self.hyp.box * distill_loss_box  # box gain
                loss[1] = self.hyp.cls * distill_loss_cls  # cls gain
                loss[2] = self.hyp.dfl * distill_loss_dfl  # dfl gain

                distill_sum = loss.sum() * self.batch_size
                total = loss_student_sum + self.kd_weight * distill_sum
                loss_out = loss_student + self.kd_weight * loss

                if self.kd_log_interval > 0:
                    self._kd_log_counter += 1
                    if self._kd_log_counter % self.kd_log_interval == 0:
                        student_cls = loss_student[1].detach().clamp_min(1e-12)
                        kd_cls = (self.kd_weight * loss[1].detach()).clamp_min(0.0)
                        ratio = (kd_cls / student_cls).item()
                        LOGGER.info(
                            f"KD Ratio | kd*L_cls_kd / L_cls_student = {ratio:.4f} "
                            f"(gate={self.teacher_conf_gate:.2f}, mode={self.cls_kd_mode})"
                        )

                return total, loss_out.detach()


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # Note: this loop follows the original Ultralytics batching logic.
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    def __init__(self, model):
        """
        Initializes v8OBBLoss with model, assigner, and rotated bbox loss.

        Note model must be de-paralleled.
        """
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)
