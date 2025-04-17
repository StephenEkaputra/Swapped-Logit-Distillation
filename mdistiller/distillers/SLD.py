from termios import CEOL
from turtle import st
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ._base import Distiller
from .loss import CrossEntropyLabelSmooth
from os.path import exists
import os


def swap(logit, target):
    #swap mechanism
    swapped_logits = torch.clone(logit)
    _, max_indices = torch.max(logit, dim=1)
    swap_mask = target != max_indices
    swapped_logits[swap_mask, target[swap_mask]], swapped_logits[swap_mask, max_indices[swap_mask]] = (
    swapped_logits[swap_mask, max_indices[swap_mask]],swapped_logits[swap_mask, target[swap_mask]],)
    return swapped_logits


def kd_loss(logits_student, logits_teacher, logits_teacher2, target, epoch, temperature, reduce=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, cls_size = logits_student.shape
    gamma = 150

    losses = []

    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd = loss_kd.mean()
    loss_kd *= temperature**2
    losses.append(loss_kd)

    if epoch > gamma:
        log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher2 / temperature, dim=1)
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
        loss_kd = loss_kd.mean()
        loss_kd *= temperature**2
        losses.append(loss_kd)

    total_loss = sum(losses)
    return total_loss


class SLD(Distiller):
    """Implementation based on MLKD (CVPR 2023)"""

    def __init__(self, student, teacher, cfg):
        super(SLD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT

    def forward_train(self, image_weak, image_strong, target, **kwargs):
        logits_student_weak, _ = self.student(image_weak)
        logits_student_strong, _ = self.student(image_strong)

        with torch.no_grad():
            logits_teacher_weak, _ = self.teacher(image_weak)
            logits_teacher_strong, _ = self.teacher(image_strong)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #Teacher Swap
        logits_teacher_weak = swap(logits_teacher_weak,target)
        logits_teacher_strong = swap(logits_teacher_strong,target)

        #Student Swap
        student_logits_weak = swap(logits_student_weak,target)
        student_logits_strong = swap(logits_student_strong,target)

        # losses
        loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target))

        loss_kd_weak = self.kd_loss_weight * kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            student_logits_weak,
            target,
            kwargs["epoch"],
            1.0,
            # reduce=False
        ) + self.kd_loss_weight * kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            student_logits_weak,
            target,
            kwargs["epoch"],
            2.0,
            # reduce=False
        ) + self.kd_loss_weight * kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            student_logits_weak,
            target,
            kwargs["epoch"],
            3.0,
            # reduce=False
        ) + self.kd_loss_weight * kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            student_logits_weak,
            target,
            kwargs["epoch"],
            4.0,
            # reduce=False
        ) + self.kd_loss_weight * kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            student_logits_weak,
            target,
            kwargs["epoch"],
            5.0,
            # reduce=False
        ) + self.kd_loss_weight * kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            student_logits_weak,
            target,
            kwargs["epoch"],
            6.0,
            # reduce=False
        )
       
        loss_kd_strong = self.kd_loss_weight * kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            student_logits_strong,
            target,
            kwargs["epoch"],
            1.0,
            # reduce=False
        ) + self.kd_loss_weight * kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            student_logits_strong,
            target,
            kwargs["epoch"],
            2.0,
            # reduce=False
        ) + self.kd_loss_weight * kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            student_logits_strong,
            target,
            kwargs["epoch"],
            3.0,
            # reduce=False
        ) + self.kd_loss_weight * kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            student_logits_strong,
            target,
            kwargs["epoch"],
            4.0,
            # reduce=False
        ) + self.kd_loss_weight * kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            student_logits_strong,
            target,
            kwargs["epoch"],
            5.0,
            # reduce=False
        ) + self.kd_loss_weight * kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            student_logits_strong,
            target,
            kwargs["epoch"],
            6.0,
            # reduce=False
        )
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": (loss_kd_weak + loss_kd_strong),
        }
        return logits_student_weak, losses_dict
