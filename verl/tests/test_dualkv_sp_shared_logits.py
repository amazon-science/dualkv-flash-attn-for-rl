"""SP=1 vs SP>1 parity for DualKV shared-P-1 gather (dense_common._dualkv_shared_logits).
Launch: torchrun --nproc_per_node=<SP> this.py"""
import torch, torch.distributed as dist
from verl.models.transformers.dense_common import _dualkv_shared_logits
from verl.utils.ulysses import set_ulysses_sequence_parallel_group

def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    hidden, vocab, temp = 64, 128, 1.0
    groups = [(40, 12, 4), (24, 8, 3), (56, 16, 2)]
    positions, off = [], 0
    for P, R, n in groups:
        positions.append(off + P - 1); off += P + R * n
    total = off

    # Build the SAME full tensors on every rank (seed on CPU then move -> device-independent).
    g = torch.Generator().manual_seed(0)
    full_hs = torch.randn(total, hidden, generator=g).to(dev)
    W = torch.randn(vocab, hidden, generator=g).to(dev)

    # Reference: SP=1 path (group unset).
    set_ulysses_sequence_parallel_group(None)
    ref = _dualkv_shared_logits(full_hs, W, positions, temp)

    # SP>1: set WORLD as SP group, pad+slice the full stream EXACTLY like slice_input_tensor
    # (contiguous chunk), call the fn on this rank's shard.
    sp_group = dist.new_group(ranks=list(range(world)))
    set_ulysses_sequence_parallel_group(sp_group)
    pad = (world - total % world) % world
    padded = full_hs if pad == 0 else torch.cat([full_hs, full_hs.new_zeros(pad, hidden)], 0)
    parts = padded.size(0) // world
    local_hs = padded[rank * parts:(rank + 1) * parts].contiguous()
    got = _dualkv_shared_logits(local_hs, W, positions, temp)

    ok = torch.allclose(ref, got, atol=1e-5, rtol=1e-5)
    maxdiff = (ref - got).abs().max().item()
    t = torch.tensor([1 if ok else 0], device=dev); dist.all_reduce(t, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"[SP={world}] shared_logits parity: {'PASS' if t.item()==1 else 'FAIL'} "
              f"(max|delta|={maxdiff:.2e}, parts={parts}, positions={positions})", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
