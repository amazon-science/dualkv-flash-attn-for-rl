#!/usr/bin/env python3
"""Closed-form peak GPU memory predictor for Qwen3-family models under
DualKV and FA2 packing, FSDP2, and block-level gradient checkpointing.

The predictor is the analytical baseline used in Appendix B. The formula has
five terms; each is closed-form in the model config, the workload config,
and the distribution settings. Comparing the predicted peak to a measured
peak evaluates whether a DualKV implementation is memory-efficient or
whether there's unexplained overhead.

Usage:
    from predict_memory import ModelConfig, WorkloadConfig, predict
    cfg  = QWEN3_8B
    work = WorkloadConfig(path="dualkv", N=8, P=16384, R=2048,
                          world_size=8, optimizer=None, dtype_bytes=2)
    pred = predict(cfg, work)
    for name, gb in pred.items():
        print(f"{name:<30} {gb:>7.2f} GB")
"""

from dataclasses import dataclass
from typing import Literal, Optional


# --------------------------------------------------------------------------- #
#                                 Config                                      #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ModelConfig:
    name: str
    L: int          # num_hidden_layers
    H: int          # num_attention_heads
    H_k: int        # num_key_value_heads
    d: int          # head_dim
    D: int          # hidden_size
    d_ff: int       # intermediate_size
    vocab: int      # vocab_size
    tie_word_embeddings: bool

    def param_count(self) -> int:
        """Approximate parameter count (ignoring biases and norm scales)."""
        # Attention: Q, K, V, O projections
        params_attn = self.D * (self.H * self.d) \
                    + 2 * self.D * (self.H_k * self.d) \
                    + (self.H * self.d) * self.D
        # MLP (SwiGLU): gate, up, down
        params_mlp = 3 * self.D * self.d_ff
        # Per-layer
        params_per_layer = params_attn + params_mlp
        # Embeddings + lm_head
        embed = self.D * self.vocab
        lm_head = 0 if self.tie_word_embeddings else self.D * self.vocab
        return self.L * params_per_layer + embed + lm_head

    def largest_fsdp_unit_params(self) -> int:
        """Bytes to all-gather for the single biggest FSDP2-wrapped unit.

        We assume per-layer wrapping (Qwen3DecoderLayer), which is the
        default auto-wrap policy used in the benchmark.
        """
        params_attn = self.D * (self.H * self.d) \
                    + 2 * self.D * (self.H_k * self.d) \
                    + (self.H * self.d) * self.D
        params_mlp = 3 * self.D * self.d_ff
        return params_attn + params_mlp


@dataclass(frozen=True)
class WorkloadConfig:
    path: Literal["fa2", "dualkv"]
    N: int                       # rollouts per prompt
    P: int                       # prompt length
    R: int                       # response length (per sequence)
    world_size: int              # FSDP2 ranks
    optimizer: Optional[str] = None   # None or "adamw" (2 fp32 moments = 8 B/param)
    dtype_bytes: int = 2         # bf16 or fp16
    fp32_grad_reduce: bool = True   # FSDP2 "reduce_dtype=fp32" keeps fp32 grads

    def T(self) -> int:
        """Packed token count per micro-batch under this packing."""
        if self.path == "fa2":
            return self.N * (self.P + self.R)
        elif self.path == "dualkv":
            return self.P + self.N * self.R
        else:
            raise ValueError(f"unknown path {self.path}")


# --------------------------------------------------------------------------- #
#                              Predefined                                     #
# --------------------------------------------------------------------------- #

QWEN3_8B = ModelConfig(
    name="Qwen3-8B",
    L=36, H=32, H_k=8, d=128, D=4096, d_ff=12288, vocab=151936,
    tie_word_embeddings=False,
)

QWEN3_14B = ModelConfig(
    name="Qwen3-14B",
    L=40, H=40, H_k=8, d=128, D=5120, d_ff=17408, vocab=151936,
    tie_word_embeddings=False,
)

QWEN3_32B = ModelConfig(
    name="Qwen3-32B",
    L=64, H=64, H_k=8, d=128, D=5120, d_ff=25600, vocab=151936,
    tie_word_embeddings=False,
)

QWEN3_FAMILY = [QWEN3_8B, QWEN3_14B, QWEN3_32B]


# --------------------------------------------------------------------------- #
#                               Predictor                                     #
# --------------------------------------------------------------------------- #

GB = 1024 ** 3


def per_token_activation_bytes(m: ModelConfig, dtype_bytes: int = 2) -> int:
    """Per-token bf16 activation footprint across one transformer block
    (sum of intermediate activations saved for backward under a no-recompute
    policy). Used in the within-block scratch term of the peak model.

    Rows match the per-token shapes in App B Table 8 for Qwen3-8B:
      pre-attn RMSNorm in : D
      Q, K, V proj (pre-RoPE) : (H + 2*H_k) * d
      attn output (pre-W_O): H * d
      residual 1          : D
      pre-MLP RMSNorm in  : D
      MLP gate + up       : 2 * d_ff
      SwiGLU intermediate : d_ff
      MLP down output     : D
      residual 2          : D
    """
    return dtype_bytes * (
        m.D + (m.H + 2 * m.H_k) * m.d + m.H * m.d
        + m.D + m.D + 2 * m.d_ff + m.d_ff + m.D + m.D
    )


def predict(model: ModelConfig, work: WorkloadConfig) -> dict:
    """Return predicted peak-memory terms (in GB) for the given config.

    Five terms; see App B for derivation:
      - model_state: sharded params + (optionally) gradients + optimizer
      - fsdp_gather: transient all-gather of the single largest FSDP unit
      - retained_acts: block-input activations kept across all L layers under
                       block-level gradient checkpointing
      - within_block_scratch: activations of one layer re-materialized during
                              that layer's backward recompute
      - lm_head_logits: output of the final LM head projection

    Predicted peak = state + gather + retained + max(within_block, lm_head)
    """
    T = work.T()
    n_params = model.param_count()

    # 1. Model state. Under FSDP2 full-shard, params live sharded across
    #    world_size ranks. Gradients match param size. Mixed-precision policy
    #    with reduce_dtype=fp32 keeps a fp32 gradient copy during reduction.
    params_sharded = n_params * work.dtype_bytes / work.world_size
    grads_sharded = n_params * (4 if work.fp32_grad_reduce else work.dtype_bytes) / work.world_size
    if work.optimizer == "adamw":
        opt_state = n_params * 8 / work.world_size   # fp32 momentum + variance
    elif work.optimizer is None:
        opt_state = 0
    else:
        raise ValueError(f"unknown optimizer {work.optimizer}")
    model_state = params_sharded + grads_sharded + opt_state

    # 2. FSDP gather surge. One layer's full (unsharded) params are temporarily
    #    gathered for its forward/backward.
    fsdp_gather = model.largest_fsdp_unit_params() * work.dtype_bytes

    # 3. Retained activations. Block-level gradient checkpointing with
    #    use_reentrant=False keeps only the block-input tensor (shape T x D)
    #    per layer.
    retained_acts = model.D * T * work.dtype_bytes * model.L

    # 4. Within-block scratch during backward recompute. The block being
    #    backward'd has its intermediate activations re-materialized and
    #    briefly live.
    c_layer = per_token_activation_bytes(model, work.dtype_bytes)
    within_block = c_layer * T

    # 5. lm_head output logits + its backward gradient. The forward logits
    #    tensor and the dlogits tensor co-exist briefly at the fwd-to-bwd
    #    boundary, doubling this term.
    lm_head_logits = 2 * T * model.vocab * work.dtype_bytes

    # 6. Framework reserved overhead. PyTorch's CUDA caching allocator holds
    #    ~2-3 GB per rank as baseline reserved memory; NCCL reserves comm
    #    buffers; FSDP2 prefetches the next layer (one extra gather). We
    #    aggregate these as a flat per-rank overhead.
    framework_overhead = 3.0 * GB + fsdp_gather  # one extra layer gather for prefetch

    # Peak: state + gather + retained + framework, plus the larger of
    # within-block-scratch and lm_head-logits (not simultaneous — lm_head is
    # live at the fwd-to-bwd boundary, within_block is live during each
    # block's recompute).
    transient_peak = max(within_block, lm_head_logits)
    peak = (model_state + fsdp_gather + retained_acts
            + transient_peak + framework_overhead)

    return {
        "T_tokens": T,
        "model_state_gb": model_state / GB,
        "fsdp_gather_gb": fsdp_gather / GB,
        "retained_acts_gb": retained_acts / GB,
        "within_block_scratch_gb": within_block / GB,
        "lm_head_logits_gb": lm_head_logits / GB,
        "transient_peak_gb": transient_peak / GB,
        "framework_overhead_gb": framework_overhead / GB,
        "peak_gb": peak / GB,
    }


# --------------------------------------------------------------------------- #
#                               CLI                                           #
# --------------------------------------------------------------------------- #

def _print_row(name: str, pred: dict) -> None:
    print(
        f"{name:<26} {pred['T_tokens']:>10,d}   "
        f"{pred['model_state_gb']:>6.2f}   "
        f"{pred['fsdp_gather_gb']:>5.2f}   "
        f"{pred['retained_acts_gb']:>6.2f}   "
        f"{pred['within_block_scratch_gb']:>6.2f}   "
        f"{pred['lm_head_logits_gb']:>7.2f}   "
        f"{pred['framework_overhead_gb']:>5.2f}   "
        f"{pred['peak_gb']:>6.2f}"
    )


def _print_table(rows):
    hdr = (f"{'Config':<26} {'T tokens':>10}   "
           f"{'state':>6}   {'gather':>5}   "
           f"{'retain':>6}   {'scratch':>6}   "
           f"{'logits':>7}   {'fwk':>5}   "
           f"{'PEAK':>6}   (all GB)")
    print(hdr)
    print("-" * len(hdr))
    for name, pred in rows:
        _print_row(name, pred)


def _demo():
    """Run the predictor across the Qwen3 family.

    Two validation points:
      1. Kernel benchmark (no optimizer): Qwen3-8B N=8 P=16384 R=2048 FSDP2(8)
         measured peak = 64.5 GB. Uses optimizer=None.
      2. Real training step (AdamW): Qwen3-8B LongReason mb=4 P=8192 R=2048
         measured peak = 70 GB (Table 7, §4.6).
    """
    print("=" * 115)
    print("Validation #1: Qwen3-8B N=8 P=16384 R=2048 FSDP2(8), no optimizer  "
          "[measured DualKV peak = 64.5 GB from trace; FA2 OOMs]")
    print("=" * 115)
    rows = []
    for path in ("dualkv", "fa2"):
        work = WorkloadConfig(path=path, N=8, P=16384, R=2048,
                              world_size=8, optimizer=None, dtype_bytes=2)
        rows.append((f"Qwen3-8B {path}", predict(QWEN3_8B, work)))
    _print_table(rows)

    print()
    print("=" * 115)
    print("Validation #2: Qwen3-8B LongReason mb=4 P=8192 R=2048 FSDP2(8) AdamW  "
          "[measured DualKV peak = 70 GB from Table 7; FA2 = 81 GB]")
    print("=" * 115)
    rows = []
    for path in ("dualkv", "fa2"):
        work = WorkloadConfig(path=path, N=4, P=8192, R=2048,
                              world_size=8, optimizer="adamw", dtype_bytes=2)
        rows.append((f"Qwen3-8B {path} adamw", predict(QWEN3_8B, work)))
    _print_table(rows)

    print()
    print("=" * 115)
    print("Extrapolation: Qwen3 family at N=8, P=16384, R=2048, FSDP2(8), AdamW")
    print("=" * 115)
    rows = []
    for model in QWEN3_FAMILY:
        for path in ("dualkv", "fa2"):
            work = WorkloadConfig(path=path, N=8, P=16384, R=2048,
                                  world_size=8, optimizer="adamw", dtype_bytes=2)
            rows.append((f"{model.name} {path} adamw", predict(model, work)))
    _print_table(rows)


if __name__ == "__main__":
    _demo()
