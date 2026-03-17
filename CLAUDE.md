# Megatron-LM (with DiLoCo + Fault Tolerance)

This is a fork of NVIDIA's Megatron-LM extended with cross-datacenter fault-tolerant training via DiLoCo and torchft.

## Project Structure

```
megatron/core/distributed/diloco.py     # DiLoCo implementation (NEW)
megatron/training/training.py           # Training loop — DiLoCo hook wired in here
megatron/training/arguments.py          # CLI args — DiLoCo args added at bottom
megatron/training/checkpointing.py      # Checkpoint helpers — DiLoCo sidecar at bottom
examples/diloco/launch_diloco.sh        # Multi-replica launch script (NEW)
tests/unit_tests/distributed/test_diloco.py  # DiLoCo unit tests (NEW)
```

## Key Architecture Decisions

- **Separate process per cluster**: each datacenter runs its own `torchrun` job; intra-replica parallelism (TP/PP/FSDP) is untouched
- **torchft Lighthouse**: standalone Rust binary coordinating cross-replica quorum
- **DiLoCo outer sync**: `post_train_step()` called after `train_step()` in the main training loop, every `--diloco-sync-every` steps
- **Checkpoint sidecar**: DiLoCo state saved to `iter_XXXXXXX/diloco_state.pt` alongside the standard Megatron checkpoint
- **HSDP already exists** in Megatron via two paths: Megatron-FSDP (`fsdp/`) and traditional DDP (`num_distributed_optimizer_instances > 1`)

## Running Tests

DiLoCo unit tests (no GPU, no distributed setup needed):

```bash
# From repo root — bypass conftest.py which requires triton
python -m pytest tests/unit_tests/distributed/test_diloco.py -v --noconftest
```

The `--noconftest` flag is needed because the project-level `conftest.py` imports `triton`, which is not required for DiLoCo tests.

## Dependencies

- PyTorch 2.7+
- torchft — must be installed from source (no PyPI wheels for macOS arm64):
  ```bash
  # Prerequisites: Rust toolchain + protobuf
  brew install protobuf
  curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
  source ~/.cargo/env
  pip install -e "~/exp/torchft[dev]"
  ```

## Launching DiLoCo Training

```bash
# 1. Start the Lighthouse (one shared instance, reachable by all clusters)
python -m torchft.lighthouse --bind [::]:29510

# 2. On each cluster (run independently)
LIGHTHOUSE_ADDR=<lighthouse-host>:29510 bash examples/diloco/launch_diloco.sh <REPLICA_ID>
```

Key env vars: `MODEL_SIZE`, `TP_SIZE`, `PP_SIZE`, `DILOCO_SYNC_EVERY`, `NUM_GPUS_PER_NODE`, `DATA_PATH`, `CHECKPOINT_PATH`.

## DiLoCo Algorithm

1. Save parameter snapshot θ₀
2. Run `--diloco-sync-every` inner steps (standard Megatron AdamW)
3. Compute pseudo-gradients: δ = θ₀ − θ_H
4. Allreduce pseudo-gradients across replicas (via torchft)
5. Restore θ₀, apply outer Nesterov SGD, save new snapshot
6. Repeat

## Code Style

Follow existing Megatron conventions: snake_case, type annotations on public APIs, `logger.info/warning` for runtime messages. No docstrings needed on private helpers.
