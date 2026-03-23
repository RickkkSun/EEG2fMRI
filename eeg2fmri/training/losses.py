from __future__ import annotations

import torch
import torch.nn.functional as F


def mean_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    return F.huber_loss(prediction, target, delta=delta)


def temporal_difference_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape[1] <= 1:
        return prediction.new_tensor(0.0)
    pred_diff = prediction[:, 1:] - prediction[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    return F.l1_loss(pred_diff, target_diff)


def conditional_flow_matching_loss(
    velocity_prediction: torch.Tensor,
    velocity_target: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(velocity_prediction, velocity_target)
