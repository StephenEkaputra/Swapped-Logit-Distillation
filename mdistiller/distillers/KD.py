import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def swap(logit, target):
    #swap mechanism
    swapped_logits = torch.clone(logit)
    _, max_indices = torch.max(logit, dim=1)
    swap_mask = target != max_indices
    swapped_logits[swap_mask, target[swap_mask]], swapped_logits[swap_mask, max_indices[swap_mask]] = (
    swapped_logits[swap_mask, max_indices[swap_mask]],swapped_logits[swap_mask, target[swap_mask]],)
    return swapped_logits


def kd_loss(logits_student, logits_teacher, logits_teacher2, target, epoch, temperature):
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

class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = 4.0

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        #KD+swap mechanisms
        logits_teacher1 = swap(logits_teacher,target)
        logits_teacher2 = swap(logits_student,target)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher1, logits_teacher2, target, kwargs["epoch"], self.temperature
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
