# DiLoCo on SLURM

Two-job setup: one SLURM job per replica (one per datacenter/partition).
A shared Lighthouse coordinates quorum across replicas.

## Directory layout

```
examples/diloco/slurm/
  lighthouse.sh          # sbatch: launches the Lighthouse coordinator
  replica.sh             # sbatch: launches one Megatron replica (run once per DC)
  baseline.sh            # sbatch: plain Megatron baseline (no DiLoCo)
  setup_env.sh           # source: module loads, venv activation, path exports
```

## Quick start

### 1. Edit `setup_env.sh`
Fill in your cluster's module names, Python env path, and shared filesystem paths.

### 2. Launch the Lighthouse (once, on a login node or dedicated allocation)
```bash
sbatch lighthouse.sh
# Note the node name it lands on — you need it for LIGHTHOUSE_ADDR
```

### 3. Launch replicas (one sbatch per DC / partition)
```bash
# Replica 0 — e.g. partition gpu-cluster-a
REPLICA_ID=0 LIGHTHOUSE_ADDR=<lighthouse-node>:29510 sbatch replica.sh

# Replica 1 — e.g. partition gpu-cluster-b
REPLICA_ID=1 LIGHTHOUSE_ADDR=<lighthouse-node>:29510 sbatch replica.sh
```

### 4. (Optional) Baseline run for comparison
```bash
sbatch baseline.sh
```

## Lighthouse address
The Lighthouse must be reachable by ALL replica nodes. Options:
- Login/head node with a fixed hostname
- A dedicated 1-node allocation that writes its hostname to a shared file
  (see the `LIGHTHOUSE_ADDR_FILE` pattern in `replica.sh`)

## Checkpointing
Each replica saves to its own directory:
  `$CHECKPOINT_PATH/replica_<ID>/`

Outer optimizer (DiLoCo) state is saved alongside as `diloco_state.pt`.
