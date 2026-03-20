# Megatron-LM (with DiLoCo + Fault Tolerance)

This is a fork of NVIDIA's Megatron-LM extended with cross-datacenter fault-tolerant training via DiLoCo and torchft.

## Project Structure

```
megatron/core/distributed/diloco.py          # DiLoCo implementation
megatron/core/distributed/data_parallel_base.py  # Base DP class (finish_grad_sync patched)
megatron/training/training.py                # Training loop — DiLoCo hook wired in here
megatron/training/arguments.py              # CLI args — DiLoCo args at bottom
megatron/training/datasets/data_samplers.py # Virtual DP pool for multi-replica sampling
examples/diloco/slurm/launch_replicas.sh    # Launch both replicas (new/resume)
examples/diloco/slurm/launch_baseline.sh    # Launch baseline run (new/resume)
examples/diloco/slurm/replica.sh            # SLURM job: one DiLoCo replica
examples/diloco/slurm/test_llama3_1b.sh     # SLURM job: baseline (no DiLoCo)
examples/diloco/slurm/replica_small.sh      # SLURM job: 100M smoke-test replica
examples/diloco/slurm/lighthouse.sh         # SLURM job: torchft lighthouse
examples/diloco/slurm/setup_env.sh          # Sourced by all SLURM scripts
tests/unit_tests/distributed/test_diloco.py # DiLoCo unit tests
```

## Key Architecture Decisions

- **Separate process per cluster**: each datacenter runs its own `torchrun` job; intra-replica parallelism (TP/PP/FSDP2) is untouched
- **torchft Lighthouse**: standalone Rust binary coordinating cross-replica quorum; address auto-discovered from `${BASE_CHECKPOINT_PATH}/lighthouse_addr.txt`
- **DiLoCo outer sync**: `post_train_step()` called after `train_step()` every `--diloco-sync-every` steps
- **FSDP2 (`--use-torch-fsdp2`)**: params are DTensors; `diloco.py` uses `.to_local()` / `DTensor.from_local()` for all snapshot/pseudo-grad/step ops
- **Checkpoint naming**: runs stored under `${BASE_CHECKPOINT_PATH}/runs/${RUN_NAME}/replica_{N}/`; use `launch_replicas.sh --name` / `--resume` to manage
- **Multi-node**: 2 nodes × 4 GPUs per replica; torchrun uses `--rdzv_backend c10d` + head-node IP resolved via `srun hostname --ip-address`
- **FSDP2 requirements**: `CUDA_DEVICE_MAX_CONNECTIONS=1024`, `--no-gradient-accumulation-fusion`
- **consumed_train_samples**: multiplied by `diloco_num_replicas` to keep data sampler positions in sync across replicas

## Key Implementation Notes

### DTensor (FSDP2) in DiLoCo
With `--use-torch-fsdp2`, model params are `DTensor` objects. The `_local()` helper in `DiLoCoOuterOptimizer` extracts the local shard:
```python
@staticmethod
def _local(t): return t.to_local() if hasattr(t, 'to_local') else t
```
Used in `_save_snapshots`, `_restore_snapshots`, `compute_pseudo_gradients`, and `step()`. When setting `.grad` on a DTensor param, must wrap back with `DTensor.from_local(...)`.

### Proxy / Internet Access on Compute Nodes
Leonardo compute nodes access the internet via an SSH SOCKS5 reverse tunnel. In `setup_env.sh`:
- Waits up to 60s for the tunnel port to appear, then validates with `curl`
- Sets `HTTP_PROXY`/`HTTPS_PROXY` only (not `ALL_PROXY` — Gloo/NCCL use raw TCP)
- `GRPC_PROXY_OVERRIDE=""` prevents torchft gRPC from using the proxy
- `NO_PROXY=.leonardo.local` covers Python HTTP clients (aiohttp, requests)
- No env-var stripping needed in `diloco.py` — `NO_PROXY` is sufficient

Tunnel script on local machine: `ssh -fN -R ${port}:127.0.0.1:${port} $node`

## Running Tests

```bash
# From repo root — bypass conftest.py which requires triton
python -m pytest tests/unit_tests/distributed/test_diloco.py -v --noconftest
```

## Dependencies

- PyTorch 2.7+
- torchft — must be installed from source (no PyPI wheels for macOS arm64):
  ```bash
  brew install protobuf
  curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
  source ~/.cargo/env
  pip install -e "~/exp/torchft[dev]"
  ```

## Launching DiLoCo Training on Leonardo

All scripts are in `examples/diloco/slurm/`. Run from that directory.

### 1. Start the Lighthouse (once)

```bash
sbatch lighthouse.sh
# Auto-writes address to ${BASE_CHECKPOINT_PATH}/lighthouse_addr.txt
```

### 2. Launch replicas

```bash
bash launch_replicas.sh --name my_exp       # new run
bash launch_replicas.sh --resume my_exp     # resume
```

Checkpoint dirs: `${BASE_CHECKPOINT_PATH}/runs/${RUN_NAME}/replica_{0,1}/`

### 3. Launch baseline (no DiLoCo)

```bash
bash launch_baseline.sh --name my_baseline
bash launch_baseline.sh --resume my_baseline
```

Checkpoint dir: `${BASE_CHECKPOINT_PATH}/runs/${RUN_NAME}/baseline/`

### 4. Monitor logs

All output (all ranks interleaved with `[rankN]:` prefix) goes to one file:

```
# DiLoCo replicas
/leonardo_scratch/fast/IscrB_Decentro/mciccone/logs/diloco_replica_{N}/<JOB_ID>/train.log

# Baseline
/leonardo_scratch/fast/IscrB_Decentro/mciccone/logs/baseline/<JOB_ID>/train.log
```

```bash
# Live follow
tail -f .../train.log

# Training metrics only
grep 'elapsed time\|lm loss' train.log | grep -v NCCL

# DiLoCo syncs
grep 'sync committed\|FT-DiLoCo' train.log

# Errors
grep -i 'error\|traceback\|exception' train.log | grep -v 'NCCL\|torchft\|srun'

# Job status
squeue -u mciccone --format='%.10i %.20j %.8T %.10M %R'
sacct -j <JOB_ID> --format=JobID,State,ExitCode -n
```

### Healthy training indicators

- `FT-DiLoCo sync committed (step=N, participants=2)` — both replicas syncing
- MFU ~0.47 at normal steps, ~0.19 at sync steps (sync overhead is expected)
- Loss decreasing, no NaN/skipped iterations

### Key configurable env vars

`TP_SIZE`, `PP_SIZE`, `DILOCO_SYNC_EVERY`, `N_REPLICAS`, `TOTAL_GLOBAL_BATCH_SIZE`, `TRAIN_ITERS`, `LR`

## DiLoCo Algorithm

1. Save parameter snapshot θ₀
2. Run `--diloco-sync-every` inner steps (standard Megatron AdamW)
3. Compute pseudo-gradients: δ = θ₀ − θ_H
4. Allreduce pseudo-gradients across replicas (via torchft Gloo)
5. Restore θ₀, apply outer Nesterov SGD, save new snapshot
6. Repeat

## Leonardo HPC (CINECA) — SSH & Rsync

Always use the SSH alias `leonardo` (configured in `~/.ssh/config`), not the full hostname:

```bash
ssh leonardo

# Rsync to Leonardo (do NOT exclude .git)
rsync -av --exclude='__pycache__' --exclude='*.pyc' --exclude='*.egg-info' \
    /Users/marcociccone/exp/megatron_claude/ \
    leonardo:/leonardo/home/userexternal/mciccone/exp/Megatron-LM/
```

Remote path: `/leonardo/home/userexternal/mciccone/exp/Megatron-LM`
(symlink → `/leonardo_work/IscrB_Decentro/mciccone/exp/Megatron-LM`)

Key paths on scratch:
- Checkpoints: `/leonardo_scratch/fast/IscrB_Decentro/mciccone/checkpoints/`
- Logs: `/leonardo_scratch/fast/IscrB_Decentro/mciccone/logs/`
- Data: `/leonardo_scratch/fast/IscrB_Decentro/mciccone/data/fineweb_edu_10bt/`
- HF cache: `/leonardo_scratch/fast/IscrB_Decentro/mciccone/huggingface/`

SLURM: account `IscrB_Decentro`, partition `boost_usr_prod`.

## Code Style

Follow existing Megatron conventions: snake_case, type annotations on public APIs, `logger.info/warning` for runtime messages. No docstrings needed on private helpers. Do not add Claude co-author to commits.

## Known TODOs / Stubs

- `should_quantize` in `DiLoCoConfig` and `--diloco-should-quantize` CLI arg: stored but not yet implemented
- `use_bucketization` / `bucket_cap_mb`: wired in but not fully optimized
- MFU calculation uses hardcoded A100 312 TFLOP/s — wrong for other GPUs
