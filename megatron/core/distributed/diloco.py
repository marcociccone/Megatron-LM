# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Additions for DiLoCo / Fault-Tolerant DiLoCo integration with torchft.

"""
DiLoCo (Distributed Low-Communication) Training for Megatron-LM
================================================================

This module implements DiLoCo and fault-tolerant DiLoCo on top of Megatron-LM,
using torchft for cross-replica coordination and fault tolerance.

Architecture:
    Each Megatron-LM instance runs as an independent replica group (separate
    torchrun job). Intra-replica parallelism (TP, PP, DP, FSDP) is handled
    by Megatron as usual. Cross-replica synchronization uses torchft's
    fault-tolerant ProcessGroup and Manager.

    Lighthouse (standalone)
         |
    +----+----+----+
    |    |    |    |
    R0   R1   R2   R3    <-- replica groups (separate Megatron jobs)
    Each runs TP+PP+FSDP internally

Algorithm (DiLoCo, arXiv:2311.08105):
    1. Save parameter snapshot theta_0
    2. Run H inner steps with AdamW (standard Megatron training)
    3. Compute pseudo-gradients: delta = theta_0 - theta_H
    4. Allreduce pseudo-gradients across replicas (via torchft)
    5. Restore theta_0, apply outer optimizer (Nesterov SGD)
    6. Repeat

Fault Tolerance:
    - torchft Manager handles quorum, heartbeating, and recovery
    - On failure: should_commit() returns False -> restore parameters -> retry
    - New replicas join via live checkpoint transfer from healthy peers
    - DiLoCo is inherently elastic: works with variable replica count

References:
    - DiLoCo: https://arxiv.org/abs/2311.08105
    - Streaming DiLoCo: https://arxiv.org/abs/2501.18512
    - torchft: https://github.com/meta-pytorch/torchft
"""

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class DiLoCoConfig:
    """Configuration for DiLoCo training.

    Args:
        enabled: Whether DiLoCo is enabled.
        sync_every: Number of inner optimizer steps between outer syncs.
        outer_lr: Learning rate for the outer optimizer (Nesterov SGD).
        outer_momentum: Momentum for the outer optimizer.
        outer_nesterov: Whether to use Nesterov momentum.
        outer_weight_decay: Weight decay for the outer optimizer.
        backup_device: Device for parameter snapshots ('cpu' or 'cuda').
            Use 'cuda' when GPU memory headroom is sufficient (e.g. with FSDP2).
            Mirrors torchft train_diloco.py which uses backup_device=device.
        lighthouse_addr: Address of the torchft Lighthouse server.
        replica_id: Unique identifier for this replica group.
        min_replica_size: Minimum number of replicas for quorum.
        torchft_timeout_sec: Timeout for torchft operations in seconds.
        torchft_quorum_timeout_sec: Timeout for quorum in seconds.
        use_gloo: Use Gloo backend for cross-replica communication (recommended).
        use_nccl: Use NCCL backend for cross-replica communication.
        pin_memory: Pin CPU memory for parameter snapshots (only used when backup_device='cpu').
        use_bucketization: Coalesce pseudo-gradient allreduces into a single bucket.
            Mirrors torchft DiLoCo use_bucketization option.
        bucket_cap_mb: Bucket size in MB for bucketized allreduce (None = single bucket).
        should_quantize: Quantize pseudo-gradients before allreduce (stub; not yet wired).
    """

    enabled: bool = False
    sync_every: int = 500
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    outer_nesterov: bool = True
    outer_weight_decay: float = 0.0
    backup_device: str = "cuda"
    lighthouse_addr: str = ""
    replica_id: str = ""
    min_replica_size: int = 1
    torchft_timeout_sec: float = 60.0
    torchft_quorum_timeout_sec: float = 300.0
    use_gloo: bool = True
    use_nccl: bool = False
    pin_memory: bool = True
    use_bucketization: bool = False
    bucket_cap_mb: Optional[int] = None
    should_quantize: bool = False
    num_replicas: int = 1   # total DiLoCo replicas (for virtual DP pool)
    replica_index: int = 0  # 0-based index of this replica


def _get_all_parameters(model_chunks: List[nn.Module]) -> List[Tuple[str, nn.Parameter]]:
    """Collect all parameters from model chunks with unique keys.

    Model chunks arise from virtual pipeline parallelism (VPP) in Megatron,
    where a single rank holds multiple pipeline stages.
    """
    all_params = []
    for chunk_idx, chunk in enumerate(model_chunks):
        for name, param in chunk.named_parameters():
            if param.requires_grad:
                all_params.append((f"chunk{chunk_idx}.{name}", param))
    return all_params


class DiLoCoOuterOptimizer:
    """Outer optimizer for DiLoCo that operates on pseudo-gradients.

    This wraps a standard SGD optimizer but manages parameter snapshots
    and pseudo-gradient computation. It is decoupled from Megatron's
    inner optimizer.
    """

    def __init__(
        self,
        named_params: List[Tuple[str, nn.Parameter]],
        lr: float = 0.7,
        momentum: float = 0.9,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        backup_device: str = "cuda",
        pin_memory: bool = True,
        use_bucketization: bool = False,
        bucket_cap_mb: Optional[int] = None,
        should_quantize: bool = False,
    ):
        self._named_params = named_params
        self._backup_device = torch.device(backup_device)
        self._pin_memory = pin_memory and backup_device == "cpu"
        self._use_bucketization = use_bucketization
        self._bucket_cap_mb = bucket_cap_mb
        self._should_quantize = should_quantize

        # Create the outer SGD optimizer over the model parameters.
        # We only call step() on it during the outer sync, setting .grad
        # to the pseudo-gradients.
        params = [p for _, p in named_params]
        self._optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )

        # Parameter snapshots: theta_0 at the start of each outer step
        self._snapshots: Dict[str, torch.Tensor] = {}
        self._save_snapshots()

    @staticmethod
    def _local(t: torch.Tensor) -> torch.Tensor:
        """Return the local shard of a DTensor (FSDP2), or the tensor unchanged.

        Mirrors torchft's extract_local_tensor(): allreduce and snapshot ops
        must operate on plain tensors, not DTensors.
        """
        if hasattr(t, 'to_local'):
            return t.to_local()
        return t

    def _save_snapshots(self) -> None:
        """Save current parameters as snapshots for pseudo-gradient computation."""
        with torch.no_grad():
            for name, param in self._named_params:
                t = self._local(param.data).detach().clone().to(self._backup_device)
                if self._pin_memory and t.device == torch.device("cpu"):
                    t = t.pin_memory()
                self._snapshots[name] = t

    def _restore_snapshots(self) -> None:
        """Restore parameters from snapshots (before outer optimizer step)."""
        with torch.no_grad():
            for name, param in self._named_params:
                # _local() returns the local storage tensor of a DTensor —
                # copying into it updates the underlying DTensor in-place.
                local = self._local(param.data)
                local.copy_(self._snapshots[name].to(local.device), non_blocking=True)

    def compute_pseudo_gradients(self) -> Dict[str, torch.Tensor]:
        """Compute pseudo-gradients: delta = theta_0 - theta_current.

        Mirrors torchft local_sgd.py:_save_grads(). Operates on local shards
        so the result is a plain tensor suitable for manager.allreduce().
        """
        pseudo_grads = {}
        with torch.no_grad():
            for name, param in self._named_params:
                local_param = self._local(param.data)
                pseudo_grads[name] = self._snapshots[name].to(local_param.device) - local_param
        return pseudo_grads

    def step(self, pseudo_grads: Dict[str, torch.Tensor]) -> None:
        """Apply outer optimizer step using averaged pseudo-gradients.

        1. Restore parameters to theta_0
        2. Set .grad to the averaged pseudo-gradients
        3. Step the outer optimizer (Nesterov SGD)
        4. Save new snapshots
        """
        # Restore to theta_0
        self._restore_snapshots()

        # Set gradients for the outer optimizer.
        # For DTensor params (FSDP2), wrap the local pseudo-grad back into a
        # DTensor so SGD operates on the correct distributed structure.
        # Mirrors torchft local_sgd.py:_set_grads().
        with torch.no_grad():
            for name, param in self._named_params:
                grad = pseudo_grads[name]
                if hasattr(param.data, 'to_local'):
                    from torch.distributed.tensor import DTensor
                    local = self._local(param.data)
                    param.grad = DTensor.from_local(
                        grad.to(local.device),
                        param.device_mesh,
                        param.placements,
                        shape=param.shape,
                        stride=param.stride(),
                    )
                else:
                    param.grad = grad.to(param.device)

        # Step outer optimizer
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)

        # Save new snapshots
        self._save_snapshots()

    def state_dict(self) -> Dict[str, Any]:
        """State dict for checkpointing."""
        return {
            "optimizer": self._optimizer.state_dict(),
            "snapshots": {
                name: t.clone() for name, t in self._snapshots.items()
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict for recovery."""
        self._optimizer.load_state_dict(state_dict["optimizer"])
        for name, t in state_dict["snapshots"].items():
            if name in self._snapshots:
                self._snapshots[name].copy_(t)


class DiLoCoTrainer:
    """Manages DiLoCo training on top of Megatron-LM.

    This class is instantiated once per Megatron training job (one per replica).
    It hooks into the training loop after each inner optimizer step and
    periodically performs the DiLoCo outer synchronization.

    Usage in Megatron's training loop::

        diloco = DiLoCoTrainer(config, model_chunks, optimizer)

        while training:
            train_step(...)  # normal Megatron forward/backward/optimizer
            diloco.post_train_step(iteration)

    For fault-tolerant DiLoCo, use FaultTolerantDiLoCoTrainer instead.
    """

    def __init__(
        self,
        config: DiLoCoConfig,
        model_chunks: List[nn.Module],
        cross_replica_pg: Optional[dist.ProcessGroup] = None,
    ):
        self._config = config
        self._model_chunks = model_chunks
        self._local_step = 0

        # Collect all trainable parameters
        self._named_params = _get_all_parameters(model_chunks)
        logger.info(
            f"DiLoCo initialized with {len(self._named_params)} parameters, "
            f"sync_every={config.sync_every}"
        )

        # Outer optimizer
        self._outer_optimizer = DiLoCoOuterOptimizer(
            named_params=self._named_params,
            lr=config.outer_lr,
            momentum=config.outer_momentum,
            nesterov=config.outer_nesterov,
            weight_decay=config.outer_weight_decay,
            backup_device=config.backup_device,
            pin_memory=config.pin_memory,
            use_bucketization=config.use_bucketization,
            bucket_cap_mb=config.bucket_cap_mb,
            should_quantize=config.should_quantize,
        )

        # Cross-replica process group for pseudo-gradient allreduce.
        # In standalone mode (no torchft), this must be provided.
        self._cross_replica_pg = cross_replica_pg

    def get_dp_info(self, dp_rank: int, dp_size: int) -> Tuple[int, int]:
        """Virtual DP pool: expand DP rank/size across replicas.

        Mirrors boomtitan's FTManager.get_dp_info.
        Returns (effective_dp_size, effective_dp_rank).
        """
        n = self._config.num_replicas
        i = self._config.replica_index
        if n > 1:
            return dp_size * n, dp_size * i + dp_rank
        return dp_size, dp_rank

    def post_train_step(self, iteration: int) -> bool:
        """Called after each Megatron train_step.

        Returns True if an outer sync was performed.
        """
        self._local_step += 1
        if self._local_step >= self._config.sync_every:
            self._diloco_sync()
            self._local_step = 0
            return True
        return False

    @torch.profiler.record_function("diloco::sync")
    def _diloco_sync(self) -> None:
        """Perform DiLoCo outer synchronization."""
        # 1. Compute pseudo-gradients
        pseudo_grads = self._outer_optimizer.compute_pseudo_gradients()

        # 2. Allreduce pseudo-gradients across replicas
        self._allreduce_pseudo_grads(pseudo_grads)

        # 3. Apply outer optimizer
        self._outer_optimizer.step(pseudo_grads)

        logger.info("DiLoCo outer sync completed")

    @torch.profiler.record_function("diloco::allreduce")
    def _allreduce_pseudo_grads(
        self, pseudo_grads: Dict[str, torch.Tensor]
    ) -> None:
        """Allreduce pseudo-gradients across replicas using the cross-replica PG."""
        if self._cross_replica_pg is None:
            return  # single replica, nothing to sync

        works = []
        for name, grad in pseudo_grads.items():
            work = dist.all_reduce(
                grad, op=dist.ReduceOp.AVG, group=self._cross_replica_pg, async_op=True
            )
            works.append(work)
        for work in works:
            work.wait()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "local_step": self._local_step,
            "outer_optimizer": self._outer_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._local_step = state_dict["local_step"]
        self._outer_optimizer.load_state_dict(state_dict["outer_optimizer"])


class FaultTolerantDiLoCoTrainer:
    """DiLoCo trainer with torchft-based fault tolerance.

    This uses torchft's Manager and ProcessGroup for:
    - Quorum-based cross-replica coordination
    - Fault-tolerant allreduce of pseudo-gradients
    - Live checkpoint recovery (no stop-the-world)
    - Elastic scaling (replicas can join/leave)

    Each Megatron instance is a separate torchrun job. The Lighthouse server
    runs as a standalone process.

    Usage::

        ft_diloco = FaultTolerantDiLoCoTrainer(config, model_chunks, optimizer)

        while training:
            train_step(...)  # normal Megatron
            ft_diloco.post_train_step(iteration)
    """

    def __init__(
        self,
        config: DiLoCoConfig,
        model_chunks: List[nn.Module],
        megatron_optimizer: Any,
        opt_param_scheduler: Any = None,
    ):
        try:
            from torchft import Manager, ProcessGroupGloo, ProcessGroupBabyNCCL
            from torchft.local_sgd import DiLoCo as TorchFTDiLoCo
        except ImportError:
            raise ImportError(
                "torchft is required for fault-tolerant DiLoCo. "
                "Install via: pip install torchft-nightly"
            )

        self._config = config
        self._model_chunks = model_chunks
        self._megatron_optimizer = megatron_optimizer
        self._opt_param_scheduler = opt_param_scheduler
        self._local_step = 0

        # Collect all trainable parameters
        self._named_params = _get_all_parameters(model_chunks)

        # Create outer optimizer
        self._outer_optimizer = DiLoCoOuterOptimizer(
            named_params=self._named_params,
            lr=config.outer_lr,
            momentum=config.outer_momentum,
            nesterov=config.outer_nesterov,
            weight_decay=config.outer_weight_decay,
            backup_device=config.backup_device,
            pin_memory=config.pin_memory,
            use_bucketization=config.use_bucketization,
            bucket_cap_mb=config.bucket_cap_mb,
            should_quantize=config.should_quantize,
        )

        # Create torchft ProcessGroup and Manager.
        # Proxy handling: HTTP_PROXY/HTTPS_PROXY are set in setup_env.sh for
        # WandB/HF access. NO_PROXY=.leonardo.local and GRPC_PROXY_OVERRIDE=""
        # ensure intra-cluster traffic (lighthouse gRPC, Gloo TCP, aiohttp
        # checkpoint transfer) bypasses the proxy without needing to strip env vars.
        timeout = timedelta(seconds=config.torchft_timeout_sec)
        if config.use_nccl:
            self._ft_pg = ProcessGroupBabyNCCL(timeout=timeout)
        else:
            self._ft_pg = ProcessGroupGloo(timeout=timeout)

        self._manager = Manager(
            pg=self._ft_pg,
            load_state_dict=self._load_state_dict,
            state_dict=self._state_dict,
            min_replica_size=config.min_replica_size,
            use_async_quorum=False,  # DiLoCo requires synchronous quorum
            timeout=timeout,
            quorum_timeout=timedelta(seconds=config.torchft_quorum_timeout_sec),
            replica_id=config.replica_id,
            lighthouse_addr=config.lighthouse_addr,
        )
        # finally:
        #     os.environ.update(_saved_env)

        logger.info(
            f"FaultTolerantDiLoCoTrainer initialized: "
            f"replica_id={config.replica_id}, "
            f"sync_every={config.sync_every}, "
            f"outer_lr={config.outer_lr}, "
            f"min_replicas={config.min_replica_size}"
        )

    def _state_dict(self) -> Dict[str, Any]:
        """State dict for torchft checkpoint transport."""
        sd = {
            "local_step": self._local_step,
            "outer_optimizer": self._outer_optimizer.state_dict(),
        }
        # Include Megatron model state
        for i, chunk in enumerate(self._model_chunks):
            sd[f"model_chunk_{i}"] = chunk.state_dict()
        # Include Megatron optimizer state
        sd["megatron_optimizer"] = self._megatron_optimizer.state_dict()
        if self._opt_param_scheduler is not None:
            sd["opt_param_scheduler"] = self._opt_param_scheduler.state_dict()
        return sd

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict from torchft checkpoint transport (recovery)."""
        self._local_step = state_dict["local_step"]
        self._outer_optimizer.load_state_dict(state_dict["outer_optimizer"])
        for i, chunk in enumerate(self._model_chunks):
            key = f"model_chunk_{i}"
            if key in state_dict:
                chunk.load_state_dict(state_dict[key])
        if "megatron_optimizer" in state_dict:
            self._megatron_optimizer.load_state_dict(state_dict["megatron_optimizer"])
        if (
            self._opt_param_scheduler is not None
            and "opt_param_scheduler" in state_dict
        ):
            self._opt_param_scheduler.load_state_dict(
                state_dict["opt_param_scheduler"]
            )

    def get_dp_info(self, dp_rank: int, dp_size: int) -> Tuple[int, int]:
        """Virtual DP pool: expand DP rank/size across replicas.

        Mirrors boomtitan's FTManager.get_dp_info.
        Returns (effective_dp_size, effective_dp_rank).
        """
        n = self._config.num_replicas
        i = self._config.replica_index
        if n > 1:
            return dp_size * n, dp_size * i + dp_rank
        return dp_size, dp_rank

    def post_train_step(self, iteration: int) -> bool:
        """Called after each Megatron train_step.

        Returns True if an outer sync was performed and committed.
        """
        self._local_step += 1
        if self._local_step >= self._config.sync_every:
            committed = self._ft_diloco_sync()
            self._local_step = 0
            return committed
        return False

    @torch.profiler.record_function("ft_diloco::sync")
    def _ft_diloco_sync(self) -> bool:
        """Fault-tolerant DiLoCo outer synchronization.

        Returns True if the sync was committed successfully.
        On failure, parameters are restored to the last snapshot.
        """
        # Start quorum — coordinate with other replicas via Lighthouse
        self._manager.start_quorum()

        # Compute pseudo-gradients
        pseudo_grads = self._outer_optimizer.compute_pseudo_gradients()

        # Allreduce pseudo-gradients via torchft (fault-tolerant).
        # Two modes mirroring torchft DiLoCo:
        #   use_bucketization=True  → _allreduce_bucketized: single flat tensor allreduce
        #   use_bucketization=False → _allreduce_per_param:  one allreduce per parameter
        if self._outer_optimizer._use_bucketization:
            names = list(pseudo_grads.keys())
            grads = [pseudo_grads[n] for n in names]
            numels = [g.numel() for g in grads]
            shapes = [g.shape for g in grads]
            flat = torch.cat([g.flatten() for g in grads])
            self._manager.allreduce(flat).wait()
            # Unpack averaged flat buffer back into pseudo_grads
            offset = 0
            for name, numel, shape in zip(names, numels, shapes):
                pseudo_grads[name] = flat[offset:offset + numel].view(shape)
                offset += numel
        else:
            works = []
            for name, grad in pseudo_grads.items():
                work = self._manager.allreduce(grad)
                works.append(work)
            for work in works:
                work.wait()

        # Check if this step should be committed
        if self._manager.should_commit():
            # Success: apply outer optimizer
            self._outer_optimizer.step(pseudo_grads)
            logger.info(
                f"FT-DiLoCo sync committed "
                f"(step={self._manager.current_step()}, "
                f"participants={self._manager.num_participants()})"
            )
            return True
        else:
            # Failure: restore parameters to last snapshot
            self._outer_optimizer._restore_snapshots()
            logger.warning(
                f"FT-DiLoCo sync NOT committed — "
                f"restoring parameters to last snapshot"
            )
            return False

    @property
    def manager(self) -> Any:
        """Access the torchft Manager (for logging, metrics, etc.)."""
        return self._manager

    def shutdown(self) -> None:
        """Clean shutdown of torchft Manager."""
        if hasattr(self, "_manager"):
            self._manager.shutdown()


def create_diloco_trainer(
    config: DiLoCoConfig,
    model_chunks: List[nn.Module],
    megatron_optimizer: Any = None,
    opt_param_scheduler: Any = None,
    cross_replica_pg: Optional[dist.ProcessGroup] = None,
) -> Optional[DiLoCoTrainer]:
    """Factory function to create the appropriate DiLoCo trainer.

    If lighthouse_addr is configured, creates a FaultTolerantDiLoCoTrainer.
    Otherwise, creates a basic DiLoCoTrainer (no fault tolerance).

    Args:
        config: DiLoCo configuration.
        model_chunks: Megatron model chunks (one per VPP stage on this rank).
        megatron_optimizer: Megatron's optimizer (needed for FT checkpoint).
        opt_param_scheduler: Megatron's LR scheduler (needed for FT checkpoint).
        cross_replica_pg: Process group for cross-replica communication
            (only for non-FT mode).

    Returns:
        DiLoCoTrainer instance, or None if DiLoCo is disabled.
    """
    if not config.enabled:
        return None

    # Propagate virtual DP pool settings to config
    try:
        from megatron.training import get_args
        args = get_args()
        config.num_replicas = getattr(args, 'diloco_num_replicas', 1)
        config.replica_index = getattr(args, 'diloco_replica_index', 0)
    except Exception:
        pass  # args not available (e.g., unit tests)

    if config.lighthouse_addr:
        # Fault-tolerant mode via torchft
        return FaultTolerantDiLoCoTrainer(
            config=config,
            model_chunks=model_chunks,
            megatron_optimizer=megatron_optimizer,
            opt_param_scheduler=opt_param_scheduler,
        )
    else:
        # Basic DiLoCo without fault tolerance
        return DiLoCoTrainer(
            config=config,
            model_chunks=model_chunks,
            cross_replica_pg=cross_replica_pg,
        )


def get_diloco_dp_info(dp_rank: int, dp_size: int) -> Tuple[int, int]:
    """Standalone version of DiLoCoTrainer.get_dp_info for use in data_samplers.py.

    Mirrors boomtitan's ft_manager.get_dp_info pattern.
    Returns (effective_dp_size, effective_dp_rank) for the virtual DP pool.
    Falls back to (dp_size, dp_rank) when DiLoCo is disabled or args unavailable.
    """
    try:
        from megatron.training import get_args
        args = get_args()
        n = getattr(args, 'diloco_num_replicas', 1)
        i = getattr(args, 'diloco_replica_index', 0)
    except Exception:
        return dp_size, dp_rank
    if n > 1:
        return dp_size * n, dp_size * i + dp_rank
    return dp_size, dp_rank
