# DualKV: Shared-Prompt Flash-Attention for RL Training

Code release for *["DualKV: Shared-Prompt Flash-Attention Kernels for Efficient Policy Updates in RL Training"](https://arxiv.org/abs/2605.15422)*.

DualKV deduplicates shared prompts in GRPO/DAPO training — instead of computing attention over `N*(P+R)` tokens, it computes over `P + N*R`, yielding up to 6x kernel speedup and 2x end-to-end throughput on long-context RL workloads. This release includes the custom flash-attention kernels, veRL integration (with Ulysses Sequence Parallelism support), and scripts to reproduce all paper experiments.

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
