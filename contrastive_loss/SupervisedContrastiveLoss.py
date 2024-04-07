import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temp: list):
        super(SupConLoss, self).__init__()
        self.temperature = temp[0]
        self.base_temperature = temp[1]

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        feats_dot_feats = torch.div(
            torch.matmul(features, features.t()),
            self.temperature
        )
        logits_max, _ = torch.max(feats_dot_feats, dim=1, keepdim=True)
        logits = feats_dot_feats - logits_max.detach()

        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)
        logits_mask = 1 - torch.eye(logits.shape[0], logits.shape[1], device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss
