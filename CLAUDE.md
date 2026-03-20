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

## Launching DiLoCo Training on Leonardo

All scripts are in `examples/diloco/slurm/`.

### 1. Start the Lighthouse (once per experiment cluster)

```bash
cd examples/diloco/slurm
sbatch lighthouse.sh
# Writes address to ${BASE_CHECKPOINT_PATH}/lighthouse_addr.txt automatically
```

### 2. Launch replicas

```bash
# New run (auto-generates name like llama3_1b_20260320_143022)
bash launch_replicas.sh

# New run with explicit name
bash launch_replicas.sh --name my_exp

# Resume existing run
bash launch_replicas.sh --resume my_exp
```

`launch_replicas.sh` verifies the lighthouse is reachable, creates per-replica checkpoint dirs under
`${BASE_CHECKPOINT_PATH}/runs/${RUN_NAME}/replica_{0,1}/`, and submits both SLURM jobs.

### 3. Monitor logs

All training output (all ranks) goes to a single file per replica per job:

```
${LOG_PATH}/diloco_replica_{N}/${SLURM_JOB_ID}/train.log
```

Typical paths on Leonardo:
```
/leonardo_scratch/fast/IscrB_Decentro/mciccone/logs/diloco_replica_0/<JOB_ID>/train.log
/leonardo_scratch/fast/IscrB_Decentro/mciccone/logs/diloco_replica_1/<JOB_ID>/train.log
```

```bash
# Live follow
tail -f /leonardo_scratch/fast/IscrB_Decentro/mciccone/logs/diloco_replica_0/<JOB_ID>/train.log

# Training metrics only (filter out NCCL/torchft noise)
grep 'elapsed time\|lm loss' train.log | grep -v NCCL

# Check syncs committed
grep 'sync committed\|FT-DiLoCo' train.log

# Check for errors
grep -i 'error\|traceback\|exception' train.log | grep -v 'NCCL\|torchft\|srun'
```

### 4. Check job status

```bash
squeue -u mciccone --format='%.10i %.20j %.8T %.10M %R'
sacct -j <JOB_ID> --format=JobID,State,ExitCode -n   # after job ends
```

### Healthy training indicators

- `FT-DiLoCo sync committed (step=N, participants=2)` — both replicas syncing
- MFU ~0.47 at normal steps, ~0.19 at sync steps (sync overhead expected)
- Loss decreasing, no NaN/skipped iterations

### Key env vars

`MODEL_SIZE`, `TP_SIZE`, `PP_SIZE`, `DILOCO_SYNC_EVERY`, `N_REPLICAS`, `TOTAL_GLOBAL_BATCH_SIZE`

## DiLoCo Algorithm

1. Save parameter snapshot θ₀
2. Run `--diloco-sync-every` inner steps (standard Megatron AdamW)
3. Compute pseudo-gradients: δ = θ₀ − θ_H
4. Allreduce pseudo-gradients across replicas (via torchft)
5. Restore θ₀, apply outer Nesterov SGD, save new snapshot
6. Repeat

## Leonardo HPC (CINECA) — SSH & Rsync

Always use the SSH alias `leonardo` (configured in `~/.ssh/config`), not the full hostname:

```bash
# SSH
ssh leonardo

# Rsync to Leonardo (include .git, exclude pycache)
rsync -av --exclude='__pycache__' --exclude='*.pyc' --exclude='*.egg-info' \
    /Users/marcociccone/exp/megatron_claude/ \
    leonardo:/leonardo/home/userexternal/mciccone/exp/Megatron-LM/
```

Remote path: `/leonardo/home/userexternal/mciccone/exp/Megatron-LM`
(symlink → `/leonardo_work/IscrB_Decentro/mciccone/exp/Megatron-LM`)

## Code Style

Follow existing Megatron conventions: snake_case, type annotations on public APIs, `logger.info/warning` for runtime messages. No docstrings needed on private helpers.
