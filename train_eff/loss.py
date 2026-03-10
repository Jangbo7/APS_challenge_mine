import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    FL(p) = -alpha * (1 - p)^gamma * log(p)

    Args:
        gamma: 聚焦参数，越大对难样本关注越多，推荐 1~3，默认 2
        alpha: 类别权重，可传入 Tensor[num_classes] 或 None
        label_smoothing: 标签平滑，默认 0.1
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor = None,
        label_smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha          # [num_classes] or None
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, num_classes]  未经过 softmax 的原始输出
            targets: [B]               类别索引
        """
        num_classes = logits.size(1)

        # label smoothing
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        # log_softmax + softmax
        log_prob = F.log_softmax(logits, dim=1)             # [B, C]
        prob = torch.exp(log_prob)                          # [B, C]

        # focal weight: (1 - p_t)^gamma
        # p_t 是对应真实类别的概率
        p_t = prob.gather(1, targets.unsqueeze(1)).squeeze(1)   # [B]
        focal_weight = (1.0 - p_t) ** self.gamma                # [B]

        # cross entropy with smooth targets
        ce = -(smooth_targets * log_prob).sum(dim=1)        # [B]

        # apply focal weight
        loss = focal_weight * ce                            # [B]

        # apply alpha (类别权重)
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets)  # [B]
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss