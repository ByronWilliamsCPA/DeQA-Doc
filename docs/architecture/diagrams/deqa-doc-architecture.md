# DeQA-Doc Architecture

> Last updated: 2026-03-12

## System Overview

```mermaid
graph TB
    subgraph ext["External Inputs"]
        images["Document Images<br/>(PDF pages, scans)"]
        diqa5000["DIQA-5000<br/>Human Labels<br/>(MOS + std)"]
        vquala["VQualA 2025<br/>Challenge Data"]
    end

    subgraph data_prep["Data Preparation"]
        train_json["Training JSON<br/>(conversations + level_probs<br/>+ gt_score + std)"]
        single_ds["SingleDataset<br/>(per-image)"]
        pair_ds["PairDataset<br/>(ranking pairs)"]
    end

    subgraph mplug["Training Backend 1: mPLUG-Owl2-7B (Primary)"]
        clip["CLIP Vision Encoder"]
        abstractor["Visual Abstractor"]
        llama["LLaMA-2 Decoder"]
        subgraph deqa_loss["DeQA Loss"]
            ce_loss["Next-token CE"]
            softkl["SoftKL"]
            inlevel["In-level"]
            ranking["Ranking"]
        end
    end

    subgraph siglip["Training Backend 2: SigLIP2-IQA v2 (86M, Modal)"]
        siglip_backbone["SigLIP2-Base Backbone<br/>(768-dim, NAFlex)"]
        siglip_heads["Attention Pooling<br/>+ 3 Task Heads<br/>(overall/sharpness/color)"]
        siglip_phases["Phase 1: Head Warmup<br/>Phase 2: Full FT + PCGrad"]
        siglip_loss["NormInNorm L1<br/>+ Gaussian NLL"]
    end

    subgraph qwen["Training Backend 3: Qwen2.5-VL-7B"]
        llamafactory["LLaMA-Factory<br/>+ DeQA Patches"]
    end

    subgraph inference["Inference & Evaluation"]
        mplug_infer["mPLUG-Owl2 Inference"]
        siglip_infer["SigLIP2 Inference"]
        qwen_infer["Qwen2.5-VL Inference"]
        scorer["Scorer API"]
        metrics["Metrics:<br/>SRCC / PLCC<br/>KL / JS / Wasserstein"]
    end

    subgraph uncertainty["Uncertainty Pipeline (13 modules)"]
        ood["OOD Detector<br/>(Mahalanobis on<br/>SigLIP2 embeds)"]
        cross_val["Cross-Validator<br/>(SigLIP2 vs DeQA)"]
        fusion["7-Signal Fusion<br/>(OOD + JSD + σ² + entropy<br/>+ spread + HyperIQA++ + OCR)"]
        vlm_val["VLM Ensemble<br/>(Gemini Flash Lite<br/>+ Qwen 122B<br/>+ GPT-4.1 tiebreaker)"]
        pseudo_orch["Pseudo-Label<br/>Orchestrator"]
        active["Active Learning<br/>(BALD queue)"]
        format_out["Format Training<br/>Data (JSON)"]
    end

    subgraph vlm_eval["VLM Teacher Evaluation"]
        prompts["7 Prompt Variants"]
        openrouter["OpenRouter API Client"]
        parser["Response Parser"]
        full_eval["Full DIQA-5000<br/>Eval Suite"]
    end

    subgraph research["Research Infrastructure"]
        papers["10-Paper Series"]
        experiments["OOD Baselines<br/>Threshold Sweeps<br/>VLM Calibration"]
    end

    %% Data flow into preparation
    images --> train_json
    diqa5000 --> train_json
    vquala --> train_json
    train_json --> single_ds
    train_json --> pair_ds

    %% Data into training backends
    single_ds --> clip
    pair_ds --> deqa_loss
    clip --> abstractor --> llama --> deqa_loss

    single_ds --> siglip_backbone
    siglip_backbone --> siglip_heads --> siglip_loss
    siglip_phases -.-> siglip_loss

    single_ds --> llamafactory

    %% Training to inference (bold = checkpoint)
    mplug ==>|checkpoint| mplug_infer
    siglip ==>|checkpoint| siglip_infer
    qwen ==>|checkpoint| qwen_infer

    mplug_infer --> scorer
    mplug_infer --> metrics
    siglip_infer --> metrics
    qwen_infer --> metrics

    %% Uncertainty pipeline
    siglip_infer -->|embeddings| ood
    mplug_infer -->|DeQA preds| cross_val
    siglip_infer -->|SigLIP2 preds| cross_val

    ood --> fusion
    cross_val --> fusion
    fusion --> pseudo_orch
    pseudo_orch -->|borderline| vlm_val
    vlm_val -->|accept/veto| pseudo_orch
    pseudo_orch -->|accepted| format_out
    pseudo_orch -->|rejected| active
    format_out -.->|expand training data| train_json

    %% VLM teacher
    images --> prompts
    prompts --> openrouter --> parser --> full_eval
    full_eval -->|teacher signal| vlm_val

    %% Research
    metrics -.-> experiments
    full_eval -.-> papers

    %% Styling
    classDef extStyle fill:#DDDDDD,stroke:#999,color:#333
    classDef dataStyle fill:#FFF3E0,stroke:#E65100,color:#333
    classDef mplugStyle fill:#E8F5E9,stroke:#2E7D32,color:#333
    classDef siglipStyle fill:#E3F2FD,stroke:#1565C0,color:#333
    classDef qwenStyle fill:#FFEBEE,stroke:#C62828,color:#333
    classDef inferStyle fill:#E0F7FA,stroke:#00695C,color:#333
    classDef uncertStyle fill:#F3E5F5,stroke:#7B1FA2,color:#333
    classDef vlmStyle fill:#FFF8E1,stroke:#F57F17,color:#333
    classDef researchStyle fill:#FCE4EC,stroke:#880E4F,color:#333

    class images,diqa5000,vquala extStyle
    class train_json,single_ds,pair_ds dataStyle
    class clip,abstractor,llama,ce_loss,softkl,inlevel,ranking mplugStyle
    class siglip_backbone,siglip_heads,siglip_phases,siglip_loss siglipStyle
    class llamafactory qwenStyle
    class mplug_infer,siglip_infer,qwen_infer,scorer,metrics inferStyle
    class ood,cross_val,fusion,vlm_val,pseudo_orch,active,format_out uncertStyle
    class prompts,openrouter,parser,full_eval vlmStyle
    class papers,experiments researchStyle
```

## Source File Traceability

### Data Preparation

| Component | Source Files |
|-----------|-------------|
| SingleDataset | `DeQA-Score/src/datasets/single_dataset.py` |
| PairDataset | `DeQA-Score/src/datasets/pair_dataset.py` |
| Data utilities | `DeQA-Score/src/datasets/utils.py` |
| Training data | `Data-DeQA-Score/DIQA/metas/`, `Data-DeQA-Score/KONIQ/metas/` |

### mPLUG-Owl2 Training (Primary)

| Component | Source Files |
|-----------|-------------|
| Training loop | `DeQA-Score/src/train/train_mem.py` |
| DeQA Loss | `DeQA-Score/src/train/loss.py` |
| Custom trainer | `DeQA-Score/src/train/mplug_owl2_trainer.py` |
| Model architecture | `DeQA-Score/src/model/modeling_mplug_owl2.py` |
| Model builder | `DeQA-Score/src/model/builder.py` |
| find_prefix() | `DeQA-Score/src/model/utils.py` |
| Training scripts | `DeQA-Score/scripts/train.sh`, `scripts/train_lora.sh` |

### SigLIP2-IQA v2 (Modal Cloud)

| Component | Source Files |
|-----------|-------------|
| Model architecture | `modal/siglip2_v2_model.py` |
| Training orchestration | `modal/train_siglip2_iqa_v2.py` |
| Data loading | `modal/siglip2_v2_data.py` |
| PCGrad optimizer | `modal/pcgrad.py` |
| Configs | `modal/configs/siglip2_v2_*.yaml` |

### Qwen2.5-VL (LLaMA-Factory Patches)

| Component | Source Files |
|-----------|-------------|
| DeQA loss adaptation | `Llamafactory/src/llamafactory/train/sft/loss.py` |
| Custom trainer | `Llamafactory/src/llamafactory/train/sft/trainer.py` |
| Data collator | `Llamafactory/src/llamafactory/data/collator.py` |
| Training args | `Llamafactory/src/llamafactory/hyparams/finetuning_args.py` |

### Inference & Evaluation

| Component | Source Files |
|-----------|-------------|
| mPLUG-Owl2 inference | `DeQA-Score/src/evaluate/iqa_eval.py` |
| Qwen2.5-VL inference | `DeQA-Score/src/evaluate/iqa_eval_qwen.py` |
| Scorer API | `DeQA-Score/src/evaluate/scorer.py` |
| SRCC/PLCC | `DeQA-Score/src/evaluate/cal_plcc_srcc.py` |
| Distribution metrics | `DeQA-Score/src/evaluate/cal_distribution_gap.py` |

### Uncertainty Pipeline (13 modules)

| Component | Source Files |
|-----------|-------------|
| Orchestrator | `DeQA-Score/src/uncertainty/pseudo_label.py` |
| OOD detector | `DeQA-Score/src/uncertainty/ood_wrapper.py` |
| Cross-validator | `DeQA-Score/src/uncertainty/cross_validator.py` |
| 7-signal fusion | `DeQA-Score/src/uncertainty/fusion.py` |
| VLM ensemble validator | `DeQA-Score/src/uncertainty/vlm_validator.py` |
| Gaussian→discrete | `DeQA-Score/src/uncertainty/gaussian_to_discrete.py` |
| Discrete metrics | `DeQA-Score/src/uncertainty/discrete_metrics.py` |
| Active learning | `DeQA-Score/src/uncertainty/active_learning.py` |
| Format output | `DeQA-Score/src/uncertainty/format_training_data.py` |
| Metadata schema | `DeQA-Score/src/uncertainty/metadata_schema.py` |
| Metadata I/O | `DeQA-Score/src/uncertainty/metadata_io.py` |
| Validation | `DeQA-Score/src/uncertainty/validation.py` |
| Metadata convert | `DeQA-Score/src/uncertainty/metadata_convert.py` |

### VLM Teacher Evaluation

| Component | Source Files |
|-----------|-------------|
| Evaluation runner | `results/vlm_teacher_eval/run_eval.py` |
| API client | `results/vlm_teacher_eval/vlm_client.py` |
| Prompt templates | `results/vlm_teacher_eval/prompts.py` |
| Response parser | `results/vlm_teacher_eval/response_parser.py` |
| Full eval suite | `results/vlm_teacher_eval/full_eval/run_full_diqa_eval.py` |

### Research

| Component | Source Files |
|-----------|-------------|
| Paper generation | `research/papers/generate_all.py` |
| OOD baselines | `research/ood_baselines/` |
| VLM calibration | `research/vlm_calibration/` |
| Threshold sensitivity | `research/threshold_sensitivity/` |

## Key Domain Constants

```
Quality Levels:  [excellent, good, fair, poor, bad]
Level Indices:   [0, 1, 2, 3, 4]
Level Scores:    [5, 4, 3, 2, 1]
MOS Formula:     dot(level_probs, [5, 4, 3, 2, 1])
Level Prefix:    "The quality of the image is"

OOD Thresholds (recalibrated 2026-03-10):
  soft_reject:   55.37 (GT TPR@95%, FPR=14.6%)
  hard_reject:   61.62 (GT TPR@80%)
```
