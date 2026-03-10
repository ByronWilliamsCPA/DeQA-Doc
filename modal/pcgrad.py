"""PCGrad: Project Conflicting Gradients for multi-task learning.

Standalone implementation with gradient conflict logging. Used in Phase 2
of SigLIP2-IQA v2.0 training to mitigate negative transfer between the
three IQA dimensions (overall, sharpness, color).

Reference: Yu et al., "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.amp import GradScaler


@dataclass(frozen=True)
class PCGradStats:
    """Statistics from a single PCGrad step.

    Attributes:
        conflict_count: Total number of pairwise gradient conflicts detected.
        projection_magnitude: Sum of projection magnitudes applied.
        per_pair_conflicts: Conflict counts keyed by ``(task_i, task_j)``.
    """

    conflict_count: int = 0
    projection_magnitude: float = 0.0
    per_pair_conflicts: dict[tuple[str, str], int] = field(default_factory=dict)


def _gather_grad_vector(model: nn.Module) -> torch.Tensor:
    """Flatten all trainable gradients into a single 1D vector."""
    grads = []
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            grads.append(p.grad.detach().flatten())
    if not grads:
        return torch.tensor([], device=next(model.parameters()).device)
    return torch.cat(grads)


def _scatter_grad_vector(model: nn.Module, grad_vec: torch.Tensor) -> None:
    """Assign values from a flat gradient vector back to model parameters."""
    offset = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            numel = p.grad.numel()
            p.grad.copy_(grad_vec[offset : offset + numel].view_as(p.grad))
            offset += numel


def pcgrad_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    task_losses: dict[str, torch.Tensor],
    scaler: GradScaler | None = None,
) -> PCGradStats:
    """Perform a PCGrad optimizer step.

    Computes per-task gradients, projects conflicting pairs, sums the
    projected gradients, and performs an optimizer step.

    Args:
        model: The model being trained.
        optimizer: The optimizer (e.g. AdamW).
        task_losses: Dict mapping task name to scalar loss tensor.
            All losses must be computed from the same forward pass
            (i.e. the computation graph must be alive).
        scaler: Optional GradScaler for AMP training.

    Returns:
        Statistics about gradient conflicts in this step.
    """
    task_names = list(task_losses.keys())
    n_tasks = len(task_names)

    # Compute per-task gradient vectors
    task_grads: dict[str, torch.Tensor] = {}
    for i, name in enumerate(task_names):
        optimizer.zero_grad()
        loss = task_losses[name]
        retain = i < n_tasks - 1  # Only retain graph for non-last task
        if scaler is not None:
            scaler.scale(loss).backward(retain_graph=retain)
            scaler.unscale_(optimizer)
        else:
            loss.backward(retain_graph=retain)
        task_grads[name] = _gather_grad_vector(model)

    # Project conflicting gradients (randomize order to avoid bias)
    conflict_count = 0
    projection_magnitude = 0.0
    per_pair: dict[tuple[str, str], int] = {}

    shuffled = list(range(n_tasks))
    random.shuffle(shuffled)

    for idx_i in shuffled:
        name_i = task_names[idx_i]
        gi = task_grads[name_i]
        if gi.numel() == 0:
            continue
        for idx_j in shuffled:
            if idx_i == idx_j:
                continue
            name_j = task_names[idx_j]
            gj = task_grads[name_j]
            if gj.numel() == 0:
                continue

            dot = torch.dot(gi, gj)
            if dot < 0:
                # Project out the conflicting component
                proj_mag = dot / (gj.norm().square() + 1e-8)
                gi = gi - proj_mag * gj
                task_grads[name_i] = gi

                conflict_count += 1
                projection_magnitude += abs(proj_mag.item())
                pair_key = (name_i, name_j)
                per_pair[pair_key] = per_pair.get(pair_key, 0) + 1

    # Sum projected gradients and assign back
    summed = torch.stack(list(task_grads.values())).mean(dim=0)
    optimizer.zero_grad()

    # Set gradients from the projected+summed vector
    offset = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.numel()
            if p.grad is None:
                p.grad = summed[offset : offset + numel].view_as(p).clone()
            else:
                p.grad.copy_(summed[offset : offset + numel].view_as(p))
            offset += numel

    # Optimizer step
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    return PCGradStats(
        conflict_count=conflict_count,
        projection_magnitude=projection_magnitude,
        per_pair_conflicts=per_pair,
    )
