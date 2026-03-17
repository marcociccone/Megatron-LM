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
        lighthouse_addr: Address of the torchft Lighthouse server.
        replica_id: Unique identifier for this replica group.
        min_replica_size: Minimum number of replicas for quorum.
        torchft_timeout_sec: Timeout for torchft operations in seconds.
        torchft_quorum_timeout_sec: Timeout for quorum in seconds.
        use_gloo: Use Gloo backend for cross-replica communication (recommended).
        use_nccl: Use NCCL backend for cross-replica communication.
        pin_memory: Pin CPU memory for parameter snapshots.
    """

    enabled: bool = False
    sync_every: int = 500
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    outer_nesterov: bool = True
    outer_weight_decay: float = 0.0
    backup_device: str = "cpu"
    lighthouse_addr: str = ""
    replica_id: str = ""
    min_replica_size: int = 1
    torchft_timeout_sec: float = 60.0
    torchft_quorum_timeout_sec: float = 300.0
    use_gloo: bool = True
    use_nccl: bool = False
    pin_memory: bool = True


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
        backup_device: str = "cpu",
        pin_memory: bool = True,
    ):
        self._named_params = named_params
        self._backup_device = torch.device(backup_device)
        self._pin_memory = pin_memory and backup_device == "cpu"

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

    def _save_snapshots(self) -> None:
        """Save current parameters as snapshots for pseudo-gradient computation."""
        with torch.no_grad():
            for name, param in self._named_params:
                t = param.data.detach().clone().to(self._backup_device)
                if self._pin_memory and t.device == torch.device("cpu"):
                    t = t.pin_memory()
                self._snapshots[name] = t

    def _restore_snapshots(self) -> None:
        """Restore parameters from snapshots (before outer optimizer step)."""
        with torch.no_grad():
            for name, param in self._named_params:
                param.data.copy_(
                    self._snapshots[name].to(param.device), non_blocking=True
                )

    def compute_pseudo_gradients(self) -> Dict[str, torch.Tensor]:
        """Compute pseudo-gradients: delta = theta_0 - theta_current.

        Returns pseudo-gradients on the same device as the parameters
        (typically GPU) for allreduce.
        """
        pseudo_grads = {}
        with torch.no_grad():
            for name, param in self._named_params:
                snapshot = self._snapshots[name].to(param.device, non_blocking=True)
                pseudo_grads[name] = snapshot - param.data
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

        # Set gradients for the outer optimizer
        with torch.no_grad():
            for name, param in self._named_params:
                param.grad = pseudo_grads[name].to(param.device)

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
        )

        # Cross-replica process group for pseudo-gradient allreduce.
        # In standalone mode (no torchft), this must be provided.
        self._cross_replica_pg = cross_replica_pg

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
        )

        # Create torchft ProcessGroup
        timeout = timedelta(seconds=config.torchft_timeout_sec)
        if config.use_nccl:
            self._ft_pg = ProcessGroupBabyNCCL(timeout=timeout)
        else:
            self._ft_pg = ProcessGroupGloo(timeout=timeout)

        # Create torchft Manager
        self._manager = Manager(
            pg=self._ft_pg,
            load_state_dict=self._load_state_dict,
            state_dict=self._state_dict,
            min_replica_size=config.min_replica_size,
            use_async_quorum=False,  # DiLoCo requires synchronous quorum
            timeout=timeout,
            quorum_timeout=timedelta(seconds=config.torchft_quorum_timeout_sec),
            replica_id=config.replica_id,
        )

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

        # Allreduce pseudo-gradients via torchft (fault-tolerant)
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
