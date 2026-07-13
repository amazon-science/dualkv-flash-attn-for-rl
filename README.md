# DualKV: Shared-Prompt Flash-Attention for RL Training

Code release for *["DualKV: Shared-Prompt Flash-Attention Kernels for Efficient Policy Updates in RL Training"](https://arxiv.org/abs/2605.15422)*.

DualKV deduplicates shared prompts in GRPO/DAPO training — instead of computing attention over `N*(P+R)` tokens, it computes over `P + N*R`, yielding up to 6x kernel speedup and 2x end-to-end throughput on long-context RL workloads. This release includes the custom flash-attention kernels, veRL integration (with Ulysses Sequence Parallelism support), and scripts to reproduce all paper experiments.

## Gemma 4 Support (`gemma4-dev` branch)

> **The `main` branch reproduces the paper (Qwen3 models) and uses the stack in [Software Environment](#software-environment) below. Gemma 4 support lives on the `gemma4-dev` branch and requires a *different, separately-pinned* software environment — documented here.** Checking out `gemma4-dev` and following the `main` setup will not work.

Gemma 4 adds: a **head-dim 512** DualKV kernel (global attention layers) and **kernel-native causal sliding-window attention** (sliding layers, `W=1024`), plus the veRL integration for the hybrid `Gemma4ForCausalLM` decoder (60 layers: 50 sliding hd256 / 10 global hd512, GQA, `attention_k_eq_v`).

### ✨ Highlight: DualKV runs Gemma-4-31B GRPO end-to-end (and the model improves)

DualKV runs a full **Gemma-4-31B GRPO training run on a single 8×H200 node** — the hd512 global layers
route through the DualKV kernel (which FA2's fused path cannot serve at hd>256), with 16K-token shared
prompts and `N=16` rollouts, **SP=1, no parameter/optimizer offload**. DualKV kernel activity is
verified at the CUDA-symbol level (eBPF uprobe on `mha_dualkv_varlen_fwd/bwd`), and held-out accuracy
**improves under GRPO** — confirming the pipeline is not just running but training.

<a name="verified-run"></a>**Verified run (single 8×H200, from-scratch pip venv, `gemma4-dev`):**

| Setting | Value |
|---------|-------|
| Model | Gemma-4-31B-it (31.27B, hd512 global + hd256 sliding) |
| Task / context | LongReason, 16384 prompt / 2048 response, thinking ON |
| Rollouts / micro-batch | `N=16` / `mb=4` |
| Parallelism | rollout TP=4, **SP=1**, no param/optimizer offload |
| Duration | 3 epochs (21 steps), ~1220 s (~20 min) per step |
| Peak GPU mem (policy-update) | **~96.8 GB / 140 GB** (H200), flat across all steps |
| DualKV kernel | fwd + bwd verified active (eBPF, ~3M kernel calls) |
| **Held-out val accuracy** | **0.748 (base) → ~0.80 (peak 0.814 @ step 18)** |

Reward is a strict verifiable check: `1.0` iff the response follows the instructed format
(`"The answer is X"`, X∈A–E) **and** matches ground truth — so the policy is rewarded for
instruction-following, not just the right letter buried in prose. Extractor verified bug-free over 794
LongReason tasks (`experiments/reward_longreason.py`).

**Per-step wall-clock breakdown** (representative mid-run step; stable ±5% across steps):

| Phase | Time | Share |
|-------|------|-------|
| Total step | 1220.6 s (~20 min) | 100% |
| gen (vLLM rollout, N=16) | 301.3 s | 25% |
| **update_actor (policy update)** | **563.9 s (~9.4 min)** | **46%** |
| old_log_prob | 142.6 s | 12% |
| ref_log_prob | 143.3 s | 12% |
| testing (per-step validation) | 190.0 s | 16% |
| adv + reward | ~4 s | <1% |

DualKV accelerates the compute-bound forward/backward phases (`update_actor` + the two log-prob passes ≈
850 s); rollout `gen` is plain vLLM and unaffected. The 190 s of `testing` reflects `test_freq=1`
(validation every step, used here to plot the accuracy curve) — a normal run (`test_freq=10`) drops it,
cutting the step to ~17 min.

> **Larger batches:** `N=32`/`mb=8` also runs (policy-update peak ~90.8 GB) but needs
> `gpu_memory_utilization=0.3` (not 0.4) to leave headroom for vLLM's KV re-acquire on wake; at 0.4 it
> hits a vLLM cumem OOM at the second step's rollout. `N=16`/`mb=4` at 0.4 is the stable default above.

**Required environment** (verified end-to-end on 8×H200; these versions are co-pinned and known-good):

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.12 | |
| PyTorch | 2.11.0+cu130 | torchvision 0.26.0, torchaudio 2.11.0 |
| CUDA | 13.0 (from pip wheels) | arch `90` (H100/H200); see nvcc note below |
| flash-attn | 2.8.4 (included, with DualKV + hd512 + SWA) | built against torch 2.11 |
| veRL | 0.8.0.dev (included, with DualKV + Gemma4 integration) | |
| vLLM | 0.23.0 | loads `Gemma4ForConditionalGeneration` (needs torchvision) |
| Transformers | 5.12.1 | Gemma 4 needs ≥5.6 |
| Ray | 2.49.0 | |

> **Verified 2026-07:** Gemma4-31B DualKV GRPO runs end-to-end on a single 8×H200 node with this stack
> (TP=4 rollout, SP=1, no offload, `gpu_memory_utilization=0.4`). See [Verified run](#verified-run) below.

```bash
git checkout gemma4-dev
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130

# IMPORTANT: `pip install torch` pulls the cu13 RUNTIME but NOT the nvcc toolchain needed to build the
# kernel. Install the cu13 compiler wheels explicitly, or the flash-attn build fails with
# "No such file or directory: .../nvidia/cu13/bin/nvcc".
pip install nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 nvidia-cuda-cccl==13.0.85 \
            nvidia-cuda-nvrtc==13.0.88 nvidia-nvvm==13.0.88

# flash-attention (with DualKV hd512 + SWA kernels), built against torch 2.11 / cu13
cd flash-attention
pip install ninja numpy packaging
git clone https://github.com/NVIDIA/cutlass.git csrc/cutlass
git -C csrc/cutlass checkout 7127592069c2fe01b041e174ba4345ef9b279671   # pinned
# point the build at the pip cu13 toolkit (NV=.../site-packages/nvidia):
export CUDA_HOME=$NV/cu13 LIBRARY_PATH=$NV/cu13/lib TORCH_CUDA_ARCH_LIST=9.0a
pip install -e . --no-build-isolation   # ~8 min compile
cd ..

# veRL with DualKV + Gemma4 integration
cd verl && pip install -e . && cd ..

pip install vllm==0.23.0 transformers==5.12.1 ray==2.49.0 wandb pandas pyarrow
```

Verify the Gemma 4 kernels:

```bash
python flash-attention/tests/test_dualkv_swa_fwd.py   # sliding-window forward
python flash-attention/tests/test_dualkv_swa_bwd.py   # sliding-window backward (incl. fp32 dKc/dVc)
```

Model + e2e GRPO/DAPO scripts:

```bash
huggingface-cli download google/gemma-4-31B-it --local-dir ${WORKDIR}/models/gemma-4-31B-it
bash experiments/run_gemma4_31b_longreason_dualkv_sp1_50step_mb4.sh   # 2 nodes (16x H100)
```

> Known limitation: DualKV + Ulysses SP>1 on Gemma 4 is not yet supported (single-GPU-per-rank `SP=1` only); `mb=8` requires SP>1 and is pending that fix.

## Repository Structure

```
├── flash-attention/   # FlashAttention-2 (commit 41b2ef6) with DualKV kernels applied
├── verl/              # veRL v0.7.0 with DualKV integration applied
├── experiments/       # Benchmarks, training scripts, reward functions
├── LICENSE            # CC-BY-NC-4.0
└── THIRD_PARTY_LICENSES
```

Key implementation files:
- **Forward kernel**: `flash-attention/csrc/flash_attn/src/flash_fwd_kernel_dualkv_training.h`
- **Backward kernel**: `flash-attention/csrc/flash_attn/src/flash_bwd_kernel_dualkv_training.h`
- **Python interface**: `flash-attention/flash_attn/flash_attn_interface.py` (search for `dualkv`)
- **veRL actor integration**: `verl/verl/workers/actor/dp_actor.py` (search for `_dualkv`)
- **Attention monkey-patch + SP**: `verl/verl/models/transformers/monkey_patch.py` (DualKV + Ulysses all-to-all)
- **SP correctness test**: `experiments/test_dualkv_sp_correctness.py`

## Hardware Requirements

| Experiment | GPUs |
|------------|------|
| Kernel benchmarks (Table 1, Table 2) | 1x H100-80GB |
| Qwen3-8B end-to-end (Table 5, Table 8) | 8x H100-80GB |
| Qwen3-14B end-to-end | 8x H100-80GB |
| DAPO end-to-end (Table 7) | 8x H100-80GB |
| Qwen3-30B-A3B multi-node (Table 3) | 16x H100-80GB (2 nodes) |
| Memory scaling sweep | 1x H100-80GB |

## Software Environment

| Package | Version |
|---------|---------|
| Python | 3.12 |
| PyTorch | 2.9.0+cu128 |
| CUDA | 12.8 |
| flash-attn | 2.8.4 (included, with DualKV) |
| veRL | 0.7.0 (included, with DualKV) |
| vLLM | 0.12.0 |
| Ray | 2.55.0 |
| Transformers | 4.57.6 |

## Setup

```bash
git clone <this-repo> dualkv && cd dualkv
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
```

### Install Flash Attention (with DualKV kernels)

```bash
cd flash-attention
pip install ninja numpy packaging
git clone --depth 1 https://github.com/NVIDIA/cutlass.git csrc/cutlass
pip install -e . --no-build-isolation
cd ..
```

Verify: `python -c "from flash_attn import flash_attn_dualkv_varlen_func; print('OK')"`

### Install veRL (with DualKV integration)

```bash
cd verl
pip install -e .
cd ..
```

### (Optional) Flash Attention 3

Only needed to reproduce FA3 baseline rows in Table 5 and Table 7:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention-3
cd /tmp/flash-attention-3 && git checkout v3.0.0 && cd hopper && pip install -e .
```

Verify: `python -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')"`

### (Optional) Prefix Grouper

Only needed to reproduce the Prefix Grouper baseline in Table 2:

```bash
pip install git+https://github.com/CASIA-IVA-Lab/PrefixGrouper.git
```

### Remaining Dependencies

```bash
pip install vllm==0.12.0 ray==2.55.0 wandb pandas pyarrow
```

### Models and Data

```bash
WORKDIR=/path/to/your/workdir

# Models
huggingface-cli download Qwen/Qwen3-8B --local-dir ${WORKDIR}/models/Qwen3-8B
huggingface-cli download Qwen/Qwen3-14B --local-dir ${WORKDIR}/models/Qwen3-14B
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir ${WORKDIR}/models/Qwen3-30B-A3B

# Data
python experiments/preprocess_longreason.py --local_save_dir ${WORKDIR}/data/longreason
python experiments/preprocess_quality.py --local_save_dir ${WORKDIR}/data/quality
```

## Reproducing Experiments

Set environment before running any script:

```bash
export WORKDIR=/path/to/your/workdir
export WANDB_API_KEY=your_key   # optional, scripts fall back to console logging
```

**Notation:** `mb` = micro-batch size (prompt groups per training step), `P` = prompt length, `N` = number of responses per prompt, `R` = response length, `SP` = Ulysses sequence parallelism degree, `DP` = data parallelism degree, `FA2`/`FA3` = FlashAttention-2/3.

### Table 1: Kernel-Level Benchmarks (1x H100 or A100)

Isolated DualKV vs FA2 attention kernel timing (fwd + bwd), fp16.

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/reproduce_table1.py
```

Expected output (H100-80GB):
```
   N      P |  FA2 fwd  FA2 bwd  FA2 f+b |   DK fwd   DK bwd   DK f+b |   fwd   bwd   f+b
  28   4096 |     49.4    165.8    215.3 |     34.4     98.7    133.1 | 1.44x 1.68x 1.62x
  28  16384 |    425.0   1325.8   1750.8 |    120.1    347.6    467.7 | 3.54x 3.81x 3.74x
  16  32768 |    857.7   2645.8   3503.4 |    174.5    504.9    679.4 | 4.91x 5.24x 5.16x
  28  32768 |   1500.9   4609.0   6109.9 |    259.8    758.4   1018.2 | 5.78x 6.08x 6.00x
  16  65536 |      OOM      OOM      OOM |    454.2   1277.7   1731.8 |   inf   inf   inf
```

### Table 2: Single-Layer DualKV vs Prefix Grouper vs FA2 (1x H100)

Single Qwen3-8B decoder layer fwd+bwd with realistic response lengths.
Prefix Grouper is self-implemented (no external package needed).

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/reproduce_table2.py
```

Paper Table 2 reports configs: (P=5K, mb=32), (8K, 16), (16K, 8), (32K, 4).
The script sweeps the full P x mb grid and marks paper configs with `*`.

### Single-Step Full-Model Benchmark (8x H100)

```bash
torchrun --standalone --nproc-per-node 8 experiments/benchmark_qwen3_single_step.py \
    --model ${WORKDIR}/models/Qwen3-8B --path both
```

### Table 5: End-to-End GRPO (Qwen3-8B, 8x H100)

| Config | Script |
|--------|--------|
| FA2 mb=4 (baseline) | `bash experiments/run_qwen3_8b_longreason_fa2.sh` |
| FA3 mb=4 | `bash experiments/run_qwen3_8b_longreason_fa3.sh` |
| DualKV mb=4 | `bash experiments/run_qwen3_8b_longreason_dualkv_mb4.sh` |
| DualKV mb=8 | `bash experiments/run_qwen3_8b_longreason_dualkv_mb8.sh` |

### Table 7: End-to-End DAPO (Qwen3-8B, 8x H100)

| Config | Script |
|--------|--------|
| FA2 mb=4 | `bash experiments/run_dapo_qwen3_8b_longreason_fa2_mb4.sh` |
| FA3 mb=4 | `bash experiments/run_dapo_qwen3_8b_longreason_fa3_mb4.sh` |
| DualKV mb=4 | `bash experiments/run_dapo_qwen3_8b_longreason_dualkv_mb4.sh` |
| DualKV mb=8 | `bash experiments/run_dapo_qwen3_8b_longreason_dualkv_mb8.sh` |

### Table 3: Multi-Node MoE (Mixture of Experts) (Qwen3-30B-A3B, 16x H100, 2 nodes)

Start a 2-node Ray cluster first:

```bash
# On node 0 (head):
ray start --head --port=6379 --num-gpus=8

# On node 1 (worker):
ray start --address='<head_ip>:6379' --num-gpus=8
```

| Config | Script |
|--------|--------|
| FA2 mb=8 SP=4 | `bash experiments/run_qwen3_30b_a3b_longreason_fa2_mb8_sp4.sh` |
| FA2 mb=8 SP=2 | `bash experiments/run_qwen3_30b_a3b_longreason_fa2_mb8_sp2.sh` |
| DualKV mb=8 | `bash experiments/run_qwen3_30b_a3b_longreason_dualkv_mb8.sh` |

### Qwen3-14B Experiments

| Config | Script |
|--------|--------|
| FA2 LongReason mb=4 | `bash experiments/run_qwen3_14b_longreason_fa2_mb4.sh` |
| DualKV LongReason mb=8 | `bash experiments/run_qwen3_14b_longreason_dualkv_mb8.sh` |
| FA2 QuALITY mb=8 | `bash experiments/run_qwen3_14b_quality_fa2_mb8.sh` |
| DualKV QuALITY mb=8 | `bash experiments/run_qwen3_14b_quality_dualkv_mb8.sh` |

### Qwen3-8B QuALITY Experiments

| Config | Script |
|--------|--------|
| FA2 | `bash experiments/run_qwen3_8b_quality_fa2.sh` |
| DualKV mb=4 | `bash experiments/run_qwen3_8b_quality_dualkv_mb4.sh` |

### Memory Scaling Sweep (1x H100)

```bash
for P in 8192 16384 32768 65536 131072; do
    python experiments/generate_padded_data.py \
        --src_dir ${WORKDIR}/data/longreason \
        --out_dir ${WORKDIR}/data/longreason_padded/P${P} \
        --target_tokens $P \
        --model_path ${WORKDIR}/models/Qwen3-8B
done

python experiments/bench_long_context_sweep.py
```

### Analytical Memory Model (Appendix B)

```bash
python experiments/predict_memory.py
```

### DualKV + Ulysses Sequence Parallelism (8x H100)

DualKV composes with Ulysses SP for additional memory savings. The DualKV repack
happens before the Ulysses sequence slice, and all-to-all communication is performed
inside the attention kernel wrapper to reconstruct the full token sequence per rank.

```bash
# DualKV + SP=2 (DP=4, SP=2): combines prompt deduplication with head splitting
bash experiments/run_qwen3_8b_longreason_dualkv_mb8_sp2.sh
```

Correctness test (requires 2 GPUs):

```bash
torchrun --nproc-per-node=2 experiments/test_dualkv_sp_correctness.py
```

## Citation

```bibtex
@article{dualkv2026,
  title={DualKV: Shared-Prompt Flash-Attention Kernels for Efficient Policy Updates in RL Training},
  author={Gai, Jiading* and Zhang, Shuai* and Song, Xiang and Wang, Bernie and Karypis, George},
  journal={arXiv preprint arXiv:2605.15422},
  year={2026}
}
```

## License

This project is licensed under CC-BY-NC-4.0. See [LICENSE](LICENSE).

This project includes code derived from [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) (BSD-3-Clause) and [veRL](https://github.com/verl-project/verl) (Apache-2.0). See [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
