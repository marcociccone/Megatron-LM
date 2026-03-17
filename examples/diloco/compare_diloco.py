"""
Baseline vs DiLoCo comparison — single process, no GPUs required.

Trains a tiny GPT model on synthetic data using:
  1. Baseline:  plain AdamW for total_steps
  2. DiLoCo:    H inner AdamW steps + outer Nesterov SGD, for total_steps // H rounds
                (single replica — no cross-DC communication, validates the mechanism)

Usage:
  # Single process (CPU / MPS / CUDA)
  python examples/diloco/compare_diloco.py

  # Override defaults
  python examples/diloco/compare_diloco.py \
      --inner-steps 50 --outer-steps 20 --hidden-size 256 --device cpu
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import AdamW

# ---- Megatron path setup -----------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Single-process distributed init (required by Megatron parallel_state)
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29501")

import torch.distributed as dist

if not dist.is_initialized():
    dist.init_process_group(backend="gloo")

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.distributed.diloco import DiLoCoConfig, DiLoCoOuterOptimizer

parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
)


# ---- Synthetic dataset --------------------------------------------------

def make_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
    """Random token sequences; labels = next-token prediction."""
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)
    return tokens[:, :-1], tokens[:, 1:]   # (B, T), (B, T)


# ---- Tiny GPT model factory ---------------------------------------------

def build_model(hidden_size: int, num_layers: int, vocab_size: int, seq_len: int) -> GPTModel:
    cfg = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=max(1, hidden_size // 64),
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
    )
    return GPTModel(
        config=cfg,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
    )


# ---- Loss helper ---------------------------------------------------------

def compute_loss(model: GPTModel, tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """CE loss via GPTModel (returns per-token logits)."""
    # attention_mask shape: (1, 1, T, T) causal mask — GPTModel handles None
    position_ids = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0).expand_as(tokens)
    logits = model(tokens, position_ids, attention_mask=None, labels=labels)
    # GPTModel returns loss tensor when labels are provided
    if isinstance(logits, torch.Tensor) and logits.ndim == 0:
        return logits
    # fallback: cross-entropy over logits
    B, T, V = logits.shape
    return nn.functional.cross_entropy(logits.reshape(B * T, V), labels.reshape(B * T))


# ---- Training runs -------------------------------------------------------

def run_baseline(args, device: torch.device):
    print("\n=== BASELINE (AdamW) ===")
    torch.manual_seed(42)
    model = build_model(args.hidden_size, args.num_layers, args.vocab_size, args.seq_len).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.inner_steps * args.outer_steps
    losses = []
    t0 = time.time()

    for step in range(total_steps):
        tokens, labels = make_batch(args.batch_size, args.seq_len, args.vocab_size, device)
        opt.zero_grad()
        loss = compute_loss(model, tokens, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if (step + 1) % args.inner_steps == 0:
            outer = (step + 1) // args.inner_steps
            avg = sum(losses[-args.inner_steps:]) / args.inner_steps
            print(f"  outer={outer:3d}/{args.outer_steps}  avg_loss={avg:.4f}  "
                  f"elapsed={time.time()-t0:.1f}s")

    return losses


def run_diloco(args, device: torch.device):
    print("\n=== DiLoCo (AdamW inner + Nesterov SGD outer) ===")
    torch.manual_seed(42)
    model = build_model(args.hidden_size, args.num_layers, args.vocab_size, args.seq_len).to(device)
    inner_opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = DiLoCoConfig(
        enabled=True,
        sync_every=args.inner_steps,   # not used directly; we call step manually
        outer_lr=args.outer_lr,
        outer_momentum=args.outer_momentum,
        outer_nesterov=True,
        outer_weight_decay=0.0,
        backup_device="cpu",
        pin_memory=False,
    )
    outer_opt = DiLoCoOuterOptimizer(config=config, model_chunks=[model])

    losses = []
    t0 = time.time()

    for outer in range(args.outer_steps):
        # ---------- H inner steps ----------
        round_losses = []
        for _ in range(args.inner_steps):
            tokens, labels = make_batch(args.batch_size, args.seq_len, args.vocab_size, device)
            inner_opt.zero_grad()
            loss = compute_loss(model, tokens, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            inner_opt.step()
            round_losses.append(loss.item())

        losses.extend(round_losses)

        # ---------- DiLoCo outer step ----------
        pseudo_grads = outer_opt.compute_pseudo_gradients()
        # Single replica: no allreduce — just apply outer SGD
        outer_opt.step(pseudo_grads)

        avg = sum(round_losses) / len(round_losses)
        print(f"  outer={outer+1:3d}/{args.outer_steps}  avg_loss={avg:.4f}  "
              f"elapsed={time.time()-t0:.1f}s")

    return losses


# ---- Summary ------------------------------------------------------------

def print_summary(baseline_losses, diloco_losses, inner_steps):
    print("\n=== Summary (avg loss per outer round) ===")
    print(f"{'Round':>6}  {'Baseline':>10}  {'DiLoCo':>10}  {'Diff':>10}")
    print("-" * 44)
    n = len(baseline_losses) // inner_steps
    for i in range(n):
        s = i * inner_steps
        e = s + inner_steps
        b = sum(baseline_losses[s:e]) / inner_steps
        d = sum(diloco_losses[s:e]) / inner_steps
        print(f"  {i+1:4d}  {b:10.4f}  {d:10.4f}  {d-b:+10.4f}")


# ---- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Baseline vs DiLoCo comparison")
    parser.add_argument("--inner-steps",  type=int,   default=100,   help="H: inner steps per DiLoCo round")
    parser.add_argument("--outer-steps",  type=int,   default=10,    help="Number of outer sync rounds")
    parser.add_argument("--hidden-size",  type=int,   default=128,   help="Model hidden size")
    parser.add_argument("--num-layers",   type=int,   default=2,     help="Transformer layers")
    parser.add_argument("--vocab-size",   type=int,   default=256,   help="Vocabulary size")
    parser.add_argument("--seq-len",      type=int,   default=32,    help="Sequence length")
    parser.add_argument("--batch-size",   type=int,   default=8,     help="Micro-batch size")
    parser.add_argument("--lr",           type=float, default=3e-4,  help="Inner optimizer lr")
    parser.add_argument("--weight-decay", type=float, default=0.1,   help="Inner AdamW weight decay")
    parser.add_argument("--outer-lr",     type=float, default=0.7,   help="DiLoCo outer lr")
    parser.add_argument("--outer-momentum", type=float, default=0.9, help="DiLoCo outer momentum")
    parser.add_argument("--device",       type=str,   default="auto",
                        help="Device: auto | cpu | cuda | mps")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    total_steps = args.inner_steps * args.outer_steps
    param_count = sum(p.numel() for p in build_model(
        args.hidden_size, args.num_layers, args.vocab_size, args.seq_len).parameters())

    print(f"Device:       {device}")
    print(f"Model params: {param_count:,}")
    print(f"Total steps:  {total_steps}  ({args.outer_steps} rounds x {args.inner_steps} inner)")
    print(f"Inner lr:     {args.lr}  |  Outer lr: {args.outer_lr}  momentum: {args.outer_momentum}")

    baseline_losses = run_baseline(args, device)
    diloco_losses   = run_diloco(args, device)
    print_summary(baseline_losses, diloco_losses, args.inner_steps)


if __name__ == "__main__":
    main()
