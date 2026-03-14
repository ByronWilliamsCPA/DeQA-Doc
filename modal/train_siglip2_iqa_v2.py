"""SigLIP2-IQA v2.0 training script for Modal.

Two-phase training with Tier 1 improvements:
- Phase 1: Head warmup (backbone frozen, 10 epochs)
- Phase 2: Full fine-tuning (CosineAnnealingWarmRestarts, PCGrad, 40 epochs)

Usage:
    # Default config (DIQA-5000 only)
    uv run modal run modal/train_siglip2_iqa_v2.py

    # Custom config
    uv run modal run modal/train_siglip2_iqa_v2.py --config modal/configs/siglip2_v2_expanded.yaml

    # Resume from checkpoint
    uv run modal run modal/train_siglip2_iqa_v2.py --resume /checkpoints/epoch_15.pt

    # Detached (long-running)
    uv run modal run --detach modal/train_siglip2_iqa_v2.py
"""

from __future__ import annotations

import modal

# ============================================================================
# Modal App & Infrastructure
# ============================================================================

app = modal.App("siglip2-iqa-v2-train")

diqa_volume = modal.Volume.from_name("diqa-train-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(
    "siglip2-v2-checkpoints", create_if_missing=True
)
siglip2_v1_volume = modal.Volume.from_name("siglip2-iqa-results")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5.0",
        "torchvision>=0.20.0",
        "transformers>=4.51.0",
        "accelerate",
        "pillow",
        "pyyaml",
        "scipy",
        "wandb",
    )
    .add_local_file("modal/siglip2_v2_model.py", "/root/modal/siglip2_v2_model.py")
    .add_local_file("modal/pcgrad.py", "/root/modal/pcgrad.py")
    .add_local_file("modal/siglip2_v2_data.py", "/root/modal/siglip2_v2_data.py")
)

# ============================================================================
# Loss Functions
# ============================================================================


def norm_in_norm_loss(
    pred: "torch.Tensor", target: "torch.Tensor"
) -> "torch.Tensor":
    """NormInNorm loss: L1 between batch-normalized predictions and targets.

    Normalizes both predictions and targets to zero mean and unit variance
    within the batch, then computes L1 distance. Optimizes rank correlation.

    Args:
        pred: Predicted scores, shape ``(B,)``.
        target: Ground truth scores, shape ``(B,)``.

    Returns:
        Scalar loss tensor.
    """
    import torch.nn.functional as F

    pred_norm = (pred - pred.mean()) / (pred.std() + 1e-8)
    target_norm = (target - target.mean()) / (target.std() + 1e-8)
    return F.l1_loss(pred_norm, target_norm)


def compute_task_losses(
    outputs: dict,
    targets: dict,
    lambda_gnll: float,
) -> dict:
    """Compute per-task losses for all IQA dimensions.

    Args:
        outputs: Model outputs ``{dim: {"mu": (B,), "sigma_sq": (B,)}}``.
        targets: Ground truth ``{dim: (B,)}``.
        lambda_gnll: Weight for GaussianNLL loss component.

    Returns:
        Dict mapping dimension name to scalar loss tensor.
    """
    import torch
    import torch.nn.functional as F

    task_losses = {}
    for dim in ("overall", "sharpness", "color"):
        mu = outputs[dim]["mu"]
        sigma_sq = outputs[dim]["sigma_sq"]
        target = targets[dim]

        loss_nin = norm_in_norm_loss(mu, target)
        loss_gnll = F.gaussian_nll_loss(mu, target, sigma_sq)
        task_losses[dim] = loss_nin + lambda_gnll * loss_gnll

    return task_losses


# ============================================================================
# Validation Metrics
# ============================================================================


def compute_val_metrics(
    predictions: dict,
    targets: dict,
) -> dict:
    """Compute SRCC, PLCC, MAE per dimension and wSRCC.

    Args:
        predictions: ``{dim: list[float]}`` of predicted MOS values.
        targets: ``{dim: list[float]}`` of ground truth MOS values.

    Returns:
        Dict with per-dimension and aggregate metrics.
    """
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    metrics: dict = {}
    srccs = {}

    for dim in ("overall", "sharpness", "color"):
        pred = np.array(predictions[dim])
        gt = np.array(targets[dim])

        srcc = spearmanr(pred, gt).statistic
        plcc = pearsonr(pred, gt).statistic
        mae = np.mean(np.abs(pred - gt))

        srccs[dim] = srcc
        metrics[f"val/{dim}_srcc"] = srcc
        metrics[f"val/{dim}_plcc"] = plcc
        metrics[f"val/{dim}_mae"] = mae

    # wSRCC: VQualA MainScore formula
    metrics["val/wsrcc"] = (
        0.5 * srccs["overall"]
        + 0.25 * srccs["sharpness"]
        + 0.25 * srccs["color"]
    )

    return metrics


# ============================================================================
# Training Function
# ============================================================================


@app.function(
    image=image,
    gpu="A10",
    timeout=21600,  # 6 hours
    volumes={
        "/data": diqa_volume,
        "/checkpoints": checkpoint_volume,
        "/v1_model": siglip2_v1_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(
    config_yaml: str,
    resume_from: str | None = None,
    v1_checkpoint: str | None = None,
) -> dict:
    """Run SigLIP2-IQA v2.0 training.

    Args:
        config_yaml: YAML config string (serialized SigLIP2V2Config).
        resume_from: Path to checkpoint to resume from.
        v1_checkpoint: Path to v1.0 checkpoint for transfer learning.

    Returns:
        Dict with final metrics and checkpoint path.
    """
    import sys
    sys.path.insert(0, "/root")

    import torch
    import wandb
    from pathlib import Path
    from transformers import AutoModel, AutoProcessor

    from modal.siglip2_v2_model import (
        SigLIP2V2Config,
        SigLIP2IQAv2,
        load_v1_checkpoint,
    )
    from modal.pcgrad import pcgrad_step

    # ---- Config ----
    config = SigLIP2V2Config.from_yaml("/tmp/config.yaml")
    # Write the passed YAML to a temp file first
    Path("/tmp/config.yaml").write_text(config_yaml)
    config = SigLIP2V2Config.from_yaml("/tmp/config.yaml")

    torch.manual_seed(config.seed)
    device = "cuda"

    # ---- WandB ----
    wandb.init(
        project="siglip2-iqa-v2",
        config=config.__dict__ if hasattr(config, "__dict__") else {},
        name=f"v2-{config.max_num_patches}p-{'attn' if config.use_attention_pooling else 'mean'}"
             f"-{'pcgrad' if config.phase2_use_pcgrad else 'standard'}",
    )

    # ---- Model ----
    print("Building model...")
    backbone = AutoModel.from_pretrained(config.backbone_id)
    model = SigLIP2IQAv2(backbone, config).to(device)

    if v1_checkpoint:
        print(f"Loading v1.0 checkpoint: {v1_checkpoint}")
        missing, unexpected = load_v1_checkpoint(model, v1_checkpoint, device)
        print(f"  Missing keys: {len(missing)} (expected: attention pools)")
        print(f"  Unexpected keys: {len(unexpected)} (expected: non-IQA heads)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # ---- Data ----
    print("Building dataloaders...")
    processor = AutoProcessor.from_pretrained(config.backbone_id)

    from modal.siglip2_v2_data import build_dataloaders
    train_loader, val_loader = build_dataloaders(
        config, processor, image_root="/data", num_workers=2
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ---- Training State ----
    best_wsrcc = -1.0
    start_epoch = 0
    start_phase = 1

    if resume_from:
        print(f"Resuming from: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        start_phase = ckpt["phase"]
        best_wsrcc = ckpt.get("best_wsrcc", -1.0)
        print(f"  Resumed at epoch {start_epoch}, phase {start_phase}, best_wsrcc={best_wsrcc:.4f}")

    # ==================================================================
    # PHASE 1: Head Warmup
    # ==================================================================

    if start_phase <= 1:
        print("\n" + "=" * 60)
        print("PHASE 1: Head Warmup")
        print("=" * 60)

        # Freeze backbone
        if config.phase1_freeze_backbone:
            for p in model.backbone.parameters():
                p.requires_grad = False

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.phase1_lr,
            weight_decay=config.weight_decay,
        )

        if resume_from and start_phase == 1 and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        phase1_start = start_epoch if start_phase == 1 else 0
        for epoch in range(phase1_start, config.phase1_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                pixel_values = batch["pixel_values"].to(device)
                spatial_shapes = batch["spatial_shapes"].to(device)
                targets = {
                    dim: torch.tensor(
                        [t[dim] for t in batch["targets"]], device=device, dtype=torch.float32
                    ) if isinstance(batch["targets"], list) else batch["targets"][dim].to(device)
                    for dim in ("overall", "sharpness", "color")
                }

                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(pixel_values, spatial_shapes)
                    task_losses = compute_task_losses(outputs, targets, config.loss_lambda_gnll)
                    loss = sum(task_losses.values()) / len(task_losses)
                    loss = loss / config.gradient_accumulation

                loss.backward()

                if (batch_idx + 1) % config.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * config.gradient_accumulation
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # Validation
            val_metrics = _run_validation(model, val_loader, config, device)
            wsrcc = val_metrics["val/wsrcc"]

            print(
                f"  Phase 1 Epoch {epoch + 1}/{config.phase1_epochs}: "
                f"loss={avg_loss:.4f}, wSRCC={wsrcc:.4f}"
            )
            wandb.log({
                "phase": 1,
                "epoch": epoch,
                "train/loss": avg_loss,
                **val_metrics,
                "lr": config.phase1_lr,
            })

            if wsrcc > best_wsrcc:
                best_wsrcc = wsrcc
                _save_checkpoint(
                    model, optimizer, None, epoch, 1, best_wsrcc, config,
                    "/checkpoints/best.pt"
                )

        # Save Phase 1 final
        _save_checkpoint(
            model, optimizer, None, config.phase1_epochs - 1, 1, best_wsrcc, config,
            "/checkpoints/phase1_final.pt"
        )

    # ==================================================================
    # PHASE 2: Full Fine-Tuning
    # ==================================================================

    print("\n" + "=" * 60)
    print("PHASE 2: Full Fine-Tuning")
    print("=" * 60)

    # Unfreeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.phase2_lr,
        weight_decay=config.weight_decay,
    )

    scheduler = None
    if config.phase2_scheduler == "cosine_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.phase2_t0,
            T_mult=config.phase2_t_mult,
            eta_min=config.phase2_eta_min,
        )
    elif config.phase2_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.phase2_epochs,
            eta_min=config.phase2_eta_min,
        )

    if resume_from and start_phase == 2:
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    phase2_start = start_epoch if start_phase == 2 else 0
    for epoch in range(phase2_start, config.phase2_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_conflicts = 0
        epoch_proj_mag = 0.0
        n_batches = 0

        # For gradient accumulation with PCGrad, accumulate losses
        accum_task_losses: dict[str, list] = {"overall": [], "sharpness": [], "color": []}

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            spatial_shapes = batch["spatial_shapes"].to(device)
            targets = {
                dim: torch.tensor(
                    [t[dim] for t in batch["targets"]], device=device, dtype=torch.float32
                ) if isinstance(batch["targets"], list) else batch["targets"][dim].to(device)
                for dim in ("overall", "sharpness", "color")
            }

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(pixel_values, spatial_shapes)
                task_losses = compute_task_losses(outputs, targets, config.loss_lambda_gnll)

            if config.phase2_use_pcgrad:
                # Accumulate for PCGrad step at gradient_accumulation boundary
                for dim in task_losses:
                    accum_task_losses[dim].append(task_losses[dim])

                if (batch_idx + 1) % config.gradient_accumulation == 0:
                    # Sum accumulated losses per task
                    summed_losses = {
                        dim: sum(losses) / len(losses)
                        for dim, losses in accum_task_losses.items()
                    }
                    stats = pcgrad_step(model, optimizer, summed_losses)
                    epoch_conflicts += stats.conflict_count
                    epoch_proj_mag += stats.projection_magnitude

                    # Reset accumulation
                    accum_task_losses = {"overall": [], "sharpness": [], "color": []}
            else:
                # Standard gradient averaging
                loss = sum(task_losses.values()) / len(task_losses)
                loss = loss / config.gradient_accumulation
                loss.backward()

                if (batch_idx + 1) % config.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad()

            batch_loss = sum(l.item() for l in task_losses.values()) / len(task_losses)
            epoch_loss += batch_loss
            n_batches += 1

        if scheduler:
            scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        # Validation
        val_metrics = _run_validation(model, val_loader, config, device)
        wsrcc = val_metrics["val/wsrcc"]

        log_dict = {
            "phase": 2,
            "epoch": config.phase1_epochs + epoch,
            "train/loss": avg_loss,
            "lr": current_lr,
            **val_metrics,
        }
        if config.phase2_use_pcgrad:
            log_dict["pcgrad/conflicts"] = epoch_conflicts
            log_dict["pcgrad/projection_magnitude"] = epoch_proj_mag

        print(
            f"  Phase 2 Epoch {epoch + 1}/{config.phase2_epochs}: "
            f"loss={avg_loss:.4f}, wSRCC={wsrcc:.4f}, lr={current_lr:.2e}"
            + (f", conflicts={epoch_conflicts}" if config.phase2_use_pcgrad else "")
        )
        wandb.log(log_dict)

        # Checkpoint
        if wsrcc > best_wsrcc:
            best_wsrcc = wsrcc
            _save_checkpoint(
                model, optimizer, scheduler, epoch, 2, best_wsrcc, config,
                "/checkpoints/best.pt"
            )
            print(f"    New best wSRCC: {best_wsrcc:.4f}")

        if (epoch + 1) % 5 == 0:
            _save_checkpoint(
                model, optimizer, scheduler, epoch, 2, best_wsrcc, config,
                f"/checkpoints/phase2_epoch{epoch + 1}.pt"
            )

    # Save final
    _save_checkpoint(
        model, optimizer, scheduler, config.phase2_epochs - 1, 2, best_wsrcc, config,
        "/checkpoints/final.pt"
    )

    # Save config alongside checkpoints
    config.to_yaml("/checkpoints/config.yaml")
    checkpoint_volume.commit()

    wandb.log({"best_wsrcc": best_wsrcc})
    wandb.finish()

    print(f"\nTraining complete. Best wSRCC: {best_wsrcc:.4f}")
    return {"best_wsrcc": best_wsrcc, "checkpoint": "/checkpoints/best.pt"}


# ============================================================================
# Helpers
# ============================================================================


def _run_validation(model, val_loader, config, device) -> dict:
    """Run validation epoch and compute metrics."""
    import torch

    model.eval()
    predictions: dict = {"overall": [], "sharpness": [], "color": []}
    targets: dict = {"overall": [], "sharpness": [], "color": []}
    sigma_sqs: dict = {"overall": [], "sharpness": [], "color": []}

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            spatial_shapes = batch["spatial_shapes"].to(device)

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(pixel_values, spatial_shapes)

            for dim in ("overall", "sharpness", "color"):
                mu = outputs[dim]["mu"].cpu().tolist()
                sq = outputs[dim]["sigma_sq"].cpu().tolist()
                predictions[dim].extend(mu)
                sigma_sqs[dim].extend(sq)

                if isinstance(batch["targets"], list):
                    targets[dim].extend([t[dim] for t in batch["targets"]])
                else:
                    targets[dim].extend(batch["targets"][dim].tolist())

    metrics = compute_val_metrics(predictions, targets)

    # Add mean sigma_sq per dimension
    import numpy as np
    for dim in ("overall", "sharpness", "color"):
        metrics[f"val/{dim}_mean_sigma_sq"] = float(np.mean(sigma_sqs[dim]))

    model.train()
    return metrics


def _save_checkpoint(model, optimizer, scheduler, epoch, phase, best_wsrcc, config, path):
    """Save training checkpoint."""
    import torch
    from pathlib import Path as P
    from dataclasses import asdict

    P(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "phase": phase,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_wsrcc": best_wsrcc,
            "config": asdict(config),
        },
        path,
    )
    checkpoint_volume.commit()


# ============================================================================
# Entrypoint
# ============================================================================


@app.local_entrypoint()
def main(
    config: str = "modal/configs/siglip2_v2_diqa_only.yaml",
    resume: str | None = None,
    v1_checkpoint: str | None = None,
) -> None:
    """Launch SigLIP2-IQA v2.0 training on Modal.

    Args:
        config: Path to YAML config file.
        resume: Path to checkpoint on Modal volume to resume from.
        v1_checkpoint: Path to v1.0 checkpoint for transfer learning.
    """
    from pathlib import Path

    config_yaml = Path(config).read_text()
    print(f"Config: {config}")
    print(f"Resume: {resume}")
    print(f"V1 checkpoint: {v1_checkpoint}")

    result = train.remote(
        config_yaml=config_yaml,
        resume_from=resume,
        v1_checkpoint=v1_checkpoint,
    )
    print(f"\nResult: {result}")
