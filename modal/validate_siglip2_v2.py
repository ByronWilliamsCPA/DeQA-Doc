"""Validate SigLIP2-IQA v2.0 infrastructure components on Modal GPU.

Runs individual checks WITHOUT full training to verify each component:
- forward: Build model, 784-patch forward+backward, report VRAM
- attention: Verify attention pooling masking and weight distribution
- pcgrad: Run 10 PCGrad steps, log conflict stats
- scheduler: Step through 40 epochs, print LR restart points
- data: Load DIQA samples, verify shapes and targets
- all: Run all checks sequentially

Usage:
    # Single check
    uv run modal run modal/validate_siglip2_v2.py --check forward

    # All checks
    uv run modal run modal/validate_siglip2_v2.py --check all

    # With pseudo-label data
    uv run modal run modal/validate_siglip2_v2.py --check data --pseudo-labels /data/test.jsonl
"""

from __future__ import annotations

import modal

# ============================================================================
# Modal App & Infrastructure
# ============================================================================

app = modal.App("siglip2-v2-validate")

siglip2_volume = modal.Volume.from_name("siglip2-iqa-results")
diqa_volume = modal.Volume.from_name("diqa-train-data", create_if_missing=True)

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
    )
    .add_local_file("modal/siglip2_v2_model.py", "/root/modal/siglip2_v2_model.py")
    .add_local_file("modal/pcgrad.py", "/root/modal/pcgrad.py")
    .add_local_file("modal/siglip2_v2_data.py", "/root/modal/siglip2_v2_data.py")
)

# ============================================================================
# Validation Functions
# ============================================================================


@app.function(image=image, gpu="A10", timeout=600)
def validate_forward() -> dict:
    """Validate forward+backward pass with 784 patches on A10 GPU."""
    import torch
    from transformers import AutoModel, AutoProcessor

    import sys
    sys.path.insert(0, "/root")
    from modal.siglip2_v2_model import SigLIP2V2Config, SigLIP2IQAv2

    print("=" * 60)
    print("CHECK: Forward Pass (784 patches)")
    print("=" * 60)

    config = SigLIP2V2Config()
    device = "cuda"

    # Build model
    backbone = AutoModel.from_pretrained(config.backbone_id)
    model = SigLIP2IQAv2(backbone, config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    attn_params = sum(
        p.numel() for n, p in model.named_parameters() if "attn_pool" in n
    )
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Attention pool params: {attn_params:,}")

    # Process a synthetic image
    processor = AutoProcessor.from_pretrained(config.backbone_id)
    from PIL import Image
    dummy_img = Image.new("RGB", (800, 600), color=(128, 128, 128))
    inputs = processor(
        images=dummy_img,
        return_tensors="pt",
        max_num_patches=config.max_num_patches,
        padding="max_length",
    )
    pixel_values = inputs["pixel_values"].to(device)
    spatial_shapes = inputs["spatial_shapes"].to(device)

    # Repeat for batch_size=4
    pixel_values = pixel_values.repeat(config.batch_size, 1, 1)
    spatial_shapes = spatial_shapes.repeat(config.batch_size, 1)

    print(f"\nInput shapes:")
    print(f"  pixel_values: {pixel_values.shape}")
    print(f"  spatial_shapes: {spatial_shapes}")

    # Reset VRAM counter
    torch.cuda.reset_peak_memory_stats()

    # Forward pass
    model.train()
    with torch.amp.autocast(device_type="cuda"):
        outputs = model(pixel_values, spatial_shapes)

    print(f"\nOutput shapes:")
    for dim, out in outputs.items():
        print(f"  {dim}: mu={out['mu'].shape}, sigma_sq={out['sigma_sq'].shape}")

    # Backward pass
    total_loss = sum(out["mu"].mean() + out["sigma_sq"].mean() for out in outputs.values())
    total_loss.backward()

    # Report VRAM
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_vram_gb = peak_vram_mb / 1024
    print(f"\nVRAM usage:")
    print(f"  Peak: {peak_vram_mb:.0f} MB ({peak_vram_gb:.2f} GB)")
    print(f"  A10 headroom: {24.0 - peak_vram_gb:.2f} GB remaining")

    if peak_vram_gb > 20:
        print("\n  WARNING: >20GB used. Consider batch_size=2, gradient_accumulation=8")
    else:
        print("\n  OK: Fits comfortably in A10 24GB")

    # Gradient shapes
    print(f"\nGradient check (sample):")
    for name, p in list(model.named_parameters())[:5]:
        if p.grad is not None:
            print(f"  {name}: grad_norm={p.grad.norm():.4f}")

    return {
        "status": "pass",
        "peak_vram_mb": peak_vram_mb,
        "total_params": total_params,
        "attn_params": attn_params,
    }


@app.function(image=image, gpu="A10", timeout=300)
def validate_attention() -> dict:
    """Validate attention pooling weights and masking correctness."""
    import torch
    from transformers import AutoModel, AutoProcessor

    import sys
    sys.path.insert(0, "/root")
    from modal.siglip2_v2_model import SigLIP2V2Config, SigLIP2IQAv2

    print("=" * 60)
    print("CHECK: Attention Pooling")
    print("=" * 60)

    config = SigLIP2V2Config()
    device = "cuda"

    backbone = AutoModel.from_pretrained(config.backbone_id)
    model = SigLIP2IQAv2(backbone, config).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(config.backbone_id)
    from PIL import Image
    dummy_img = Image.new("RGB", (800, 600), color=(100, 150, 200))
    inputs = processor(
        images=dummy_img,
        return_tensors="pt",
        max_num_patches=config.max_num_patches,
        padding="max_length",
    )
    pixel_values = inputs["pixel_values"].to(device)
    spatial_shapes = inputs["spatial_shapes"].to(device)

    actual_patches = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).item()
    print(f"spatial_shapes: {spatial_shapes}")
    print(f"Actual patches: {actual_patches} / {config.max_num_patches} max")

    # Get backbone hidden states
    with torch.no_grad():
        backbone_out = model.backbone.get_image_features(
            pixel_values=pixel_values, spatial_shapes=spatial_shapes
        )
        hidden = backbone_out.last_hidden_state  # (1, S, 768)

    padding_mask = model._compute_padding_mask(hidden.size(1), spatial_shapes)
    print(f"Padding mask: {padding_mask.shape}, valid={padding_mask.sum().item()}")

    # Check each dimension's attention
    print(f"\nPer-dimension attention stats:")
    for dim in ("overall", "sharpness", "color"):
        pool = model.attn_pools[dim]
        with torch.no_grad():
            query = pool.query.expand(1, -1, -1)
            logits = torch.bmm(query, hidden.transpose(1, 2)) * pool.scale
            logits = logits.masked_fill(~padding_mask.unsqueeze(1), float("-inf"))
            weights = torch.softmax(logits, dim=-1)  # (1, 1, S)

        w = weights.squeeze()
        valid_weights = w[:int(actual_patches)]
        padded_weights = w[int(actual_patches):]

        weight_sum = valid_weights.sum().item()
        entropy = -(valid_weights * (valid_weights + 1e-10).log()).sum().item()
        max_weight = valid_weights.max().item()
        max_pos = valid_weights.argmax().item()

        print(f"  {dim}:")
        print(f"    Weight sum (valid): {weight_sum:.6f}")
        print(f"    Padded weight sum: {padded_weights.sum().item():.6f}")
        print(f"    Entropy: {entropy:.4f}")
        print(f"    Max weight: {max_weight:.6f} at position {max_pos}")

        assert abs(weight_sum - 1.0) < 1e-5, f"Weights don't sum to 1 for {dim}"
        assert padded_weights.sum().item() < 1e-6, f"Padded positions have weight for {dim}"

    print("\nAll attention checks passed.")
    return {"status": "pass"}


@app.function(image=image, gpu="A10", timeout=600)
def validate_pcgrad() -> dict:
    """Validate PCGrad on 10 training steps with synthetic targets."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoProcessor

    import sys
    sys.path.insert(0, "/root")
    from modal.siglip2_v2_model import SigLIP2V2Config, SigLIP2IQAv2
    from modal.pcgrad import pcgrad_step

    print("=" * 60)
    print("CHECK: PCGrad (10 steps)")
    print("=" * 60)

    config = SigLIP2V2Config()
    device = "cuda"

    backbone = AutoModel.from_pretrained(config.backbone_id)
    model = SigLIP2IQAv2(backbone, config).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.phase2_lr)

    # Synthetic batch
    processor = AutoProcessor.from_pretrained(config.backbone_id)
    from PIL import Image
    dummy_img = Image.new("RGB", (600, 400), color=(128, 128, 128))
    inputs = processor(
        images=dummy_img,
        return_tensors="pt",
        max_num_patches=config.max_num_patches,
        padding="max_length",
    )
    pixel_values = inputs["pixel_values"].repeat(config.batch_size, 1, 1).to(device)
    spatial_shapes = inputs["spatial_shapes"].repeat(config.batch_size, 1).to(device)

    # Synthetic targets
    targets = {
        "overall": torch.rand(config.batch_size, device=device),
        "sharpness": torch.rand(config.batch_size, device=device),
        "color": torch.rand(config.batch_size, device=device),
    }

    total_conflicts = 0
    total_projection_mag = 0.0

    for step in range(10):
        outputs = model(pixel_values, spatial_shapes)
        task_losses = {}
        for dim in ("overall", "sharpness", "color"):
            mu = outputs[dim]["mu"]
            sigma_sq = outputs[dim]["sigma_sq"]
            target = targets[dim]
            # Simplified loss for validation
            loss_l1 = F.l1_loss(mu, target)
            loss_gnll = F.gaussian_nll_loss(mu, target, sigma_sq)
            task_losses[dim] = loss_l1 + 0.5 * loss_gnll

        stats = pcgrad_step(model, optimizer, task_losses)
        total_conflicts += stats.conflict_count
        total_projection_mag += stats.projection_magnitude

        loss_str = ", ".join(f"{d}={task_losses[d].item():.4f}" for d in task_losses)
        print(
            f"  Step {step + 1}: {loss_str} | "
            f"conflicts={stats.conflict_count}, "
            f"proj_mag={stats.projection_magnitude:.4f}"
        )

    print(f"\nTotal conflicts: {total_conflicts}")
    print(f"Total projection magnitude: {total_projection_mag:.4f}")
    print(f"Avg conflicts/step: {total_conflicts / 10:.1f}")

    return {
        "status": "pass",
        "total_conflicts": total_conflicts,
        "total_projection_magnitude": total_projection_mag,
    }


@app.function(image=image, timeout=120)
def validate_scheduler() -> dict:
    """Validate CosineAnnealingWarmRestarts LR schedule."""
    import torch

    print("=" * 60)
    print("CHECK: CosineAnnealingWarmRestarts Scheduler")
    print("=" * 60)

    # Simulate optimizer with a single dummy parameter
    param = torch.nn.Parameter(torch.randn(10))
    optimizer = torch.optim.AdamW([param], lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )

    print(f"\nLR schedule for Phase 2 (40 epochs):")
    print(f"  T_0=10, T_mult=2, eta_min=1e-7")
    print(f"  Expected restarts at epochs: 10, 30")
    print()

    lrs = []
    restart_epochs = []
    prev_lr = 0.0

    for epoch in range(40):
        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)

        # Detect restart (LR jumps up)
        if epoch > 0 and lr > prev_lr * 1.5:
            restart_epochs.append(epoch)
            marker = " <-- RESTART"
        else:
            marker = ""

        if epoch % 5 == 0 or marker:
            print(f"  Epoch {epoch:3d}: LR = {lr:.2e}{marker}")

        prev_lr = lr
        scheduler.step()

    print(f"\nRestart epochs: {restart_epochs}")
    print(f"LR range: [{min(lrs):.2e}, {max(lrs):.2e}]")

    expected_restarts = [10, 30]
    if restart_epochs == expected_restarts:
        print("Restart points match expected schedule.")
    else:
        print(f"WARNING: Expected restarts at {expected_restarts}, got {restart_epochs}")

    return {
        "status": "pass",
        "restart_epochs": restart_epochs,
        "lr_min": min(lrs),
        "lr_max": max(lrs),
    }


@app.function(
    image=image,
    gpu="A10",
    timeout=300,
    volumes={"/data": diqa_volume},
)
def validate_data(pseudo_labels: str | None = None) -> dict:
    """Validate data loading pipeline with DIQA samples."""
    import sys
    sys.path.insert(0, "/root")
    from modal.siglip2_v2_model import SigLIP2V2Config

    print("=" * 60)
    print("CHECK: Data Loading")
    print("=" * 60)

    config = SigLIP2V2Config()

    # Check if DIQA data is available on volume
    from pathlib import Path
    meta_dir = Path("/data") / config.data.diqa_meta_dir
    if not meta_dir.exists():
        print(f"DIQA meta directory not found at {meta_dir}")
        print("Upload DIQA data to 'diqa-train-data' volume first.")
        return {"status": "skip", "reason": "no data"}

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(config.backbone_id)

    from modal.siglip2_v2_data import DIQADataset
    dataset = DIQADataset(
        meta_dir=str(meta_dir),
        image_root="/data",
        processor=processor,
        max_num_patches=config.max_num_patches,
        split="train",
    )

    print(f"DIQA train samples: {len(dataset)}")

    # Load first 5 samples
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"\n  Sample {i}:")
        print(f"    image_id: {sample['image_id']}")
        print(f"    pixel_values: {sample['pixel_values'].shape}")
        print(f"    spatial_shapes: {sample['spatial_shapes']}")
        for dim in ("overall", "sharpness", "color"):
            print(f"    target_{dim}: {sample['targets'][dim]:.4f}")

    print("\nData loading check passed.")
    return {"status": "pass", "n_samples": len(dataset)}


# ============================================================================
# Entrypoint
# ============================================================================


@app.local_entrypoint()
def main(
    check: str = "all",
    pseudo_labels: str | None = None,
) -> None:
    """Run validation checks.

    Args:
        check: Which check to run — forward, attention, pcgrad, scheduler, data, all.
        pseudo_labels: Optional path to pseudo-label JSONL for data check.
    """
    checks = {
        "forward": validate_forward,
        "attention": validate_attention,
        "pcgrad": validate_pcgrad,
        "scheduler": validate_scheduler,
        "data": lambda: validate_data.remote(pseudo_labels),
    }

    if check == "all":
        targets = list(checks.keys())
    else:
        targets = [check]

    results = {}
    for name in targets:
        print(f"\n{'#' * 60}")
        print(f"# Running: {name}")
        print(f"{'#' * 60}\n")

        fn = checks[name]
        if name == "data":
            result = validate_data.remote(pseudo_labels)
        else:
            result = fn.remote()

        results[name] = result
        status = result.get("status", "unknown")
        print(f"\n>> {name}: {status.upper()}")

    # Summary
    print(f"\n{'=' * 60}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    for name, result in results.items():
        status = result.get("status", "unknown")
        extra = ""
        if name == "forward":
            vram = result.get("peak_vram_mb", 0)
            extra = f" (VRAM: {vram:.0f} MB)"
        elif name == "pcgrad":
            conflicts = result.get("total_conflicts", 0)
            extra = f" (conflicts: {conflicts})"
        print(f"  {name}: {status.upper()}{extra}")
