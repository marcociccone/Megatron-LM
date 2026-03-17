# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for DiLoCo outer optimizer and trainer.

These tests do NOT require a distributed process group or torchft.
They validate the core DiLoCo algorithm (snapshot, pseudo-gradient,
outer optimizer step) and the training-loop hook logic on CPU.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from megatron.core.distributed.diloco import (
    DiLoCoConfig,
    DiLoCoOuterOptimizer,
    DiLoCoTrainer,
    _get_all_parameters,
    create_diloco_trainer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_model(hidden: int = 16) -> nn.Module:
    return nn.Sequential(
        nn.Linear(hidden, hidden, bias=False),
        nn.Linear(hidden, hidden, bias=False),
    )


def _named_params(model):
    return [(f"chunk0.{n}", p) for n, p in model.named_parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# _get_all_parameters
# ---------------------------------------------------------------------------

class TestGetAllParameters(unittest.TestCase):
    def test_single_chunk(self):
        model = _simple_model()
        params = _get_all_parameters([model])
        self.assertEqual(len(params), len(list(model.parameters())))
        for name, p in params:
            self.assertTrue(name.startswith("chunk0."))
            self.assertTrue(p.requires_grad)

    def test_multi_chunk(self):
        chunks = [_simple_model(), _simple_model()]
        params = _get_all_parameters(chunks)
        expected = sum(len(list(c.parameters())) for c in chunks)
        self.assertEqual(len(params), expected)
        self.assertTrue(any(n.startswith("chunk0.") for n, _ in params))
        self.assertTrue(any(n.startswith("chunk1.") for n, _ in params))

    def test_no_grad_params_excluded(self):
        model = _simple_model()
        for p in model.parameters():
            p.requires_grad = False
        params = _get_all_parameters([model])
        self.assertEqual(len(params), 0)


# ---------------------------------------------------------------------------
# DiLoCoOuterOptimizer
# ---------------------------------------------------------------------------

class TestDiLoCoOuterOptimizer(unittest.TestCase):
    def _make_opt(self, model, **kwargs) -> DiLoCoOuterOptimizer:
        defaults = dict(
            lr=0.7, momentum=0.9, nesterov=True,
            weight_decay=0.0, backup_device="cpu", pin_memory=False,
        )
        defaults.update(kwargs)
        return DiLoCoOuterOptimizer(_named_params(model), **defaults)

    def test_snapshots_match_initial_params(self):
        model = _simple_model()
        opt = self._make_opt(model)
        for name, param in _named_params(model):
            torch.testing.assert_close(opt._snapshots[name], param.data.cpu())

    def test_pseudo_gradients_zero_before_any_inner_step(self):
        model = _simple_model()
        opt = self._make_opt(model)
        pseudo_grads = opt.compute_pseudo_gradients()
        for grad in pseudo_grads.values():
            torch.testing.assert_close(grad, torch.zeros_like(grad))

    def test_pseudo_gradients_nonzero_after_inner_step(self):
        model = _simple_model()
        opt = self._make_opt(model)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        pseudo_grads = opt.compute_pseudo_gradients()
        nonzero = any(g.abs().max() > 0 for g in pseudo_grads.values())
        self.assertTrue(nonzero, "Expected non-zero pseudo-gradients after inner step")

    def test_restore_snapshots(self):
        model = _simple_model()
        original = {n: p.data.clone() for n, p in _named_params(model)}
        opt = self._make_opt(model)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)
        opt._restore_snapshots()
        for name, param in _named_params(model):
            torch.testing.assert_close(param.data, original[name])

    def test_step_updates_params(self):
        model = _simple_model()
        opt = self._make_opt(model)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        pseudo_grads = opt.compute_pseudo_gradients()
        params_before = {n: p.data.clone() for n, p in _named_params(model)}
        opt.step(pseudo_grads)
        any_changed = any(
            not torch.equal(p.data, params_before[n])
            for n, p in _named_params(model)
        )
        self.assertTrue(any_changed, "Expected params to change after outer optimizer step")

    def test_step_saves_new_snapshots(self):
        model = _simple_model()
        opt = self._make_opt(model)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        pseudo_grads = opt.compute_pseudo_gradients()
        opt.step(pseudo_grads)
        for name, param in _named_params(model):
            torch.testing.assert_close(opt._snapshots[name], param.data.cpu())

    def test_state_dict_round_trip(self):
        model = _simple_model()
        opt = self._make_opt(model)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        pseudo_grads = opt.compute_pseudo_gradients()
        opt.step(pseudo_grads)

        sd = opt.state_dict()
        model2 = _simple_model()
        opt2 = self._make_opt(model2)
        opt2.load_state_dict(sd)

        for name in opt._snapshots:
            torch.testing.assert_close(opt2._snapshots[name], opt._snapshots[name])


# ---------------------------------------------------------------------------
# DiLoCoTrainer
# ---------------------------------------------------------------------------

class TestDiLoCoTrainer(unittest.TestCase):
    def _make_trainer(self, sync_every=3, cross_replica_pg=None):
        model = _simple_model()
        config = DiLoCoConfig(enabled=True, sync_every=sync_every, pin_memory=False)
        trainer = DiLoCoTrainer(config, [model], cross_replica_pg=cross_replica_pg)
        return model, trainer

    def test_no_sync_before_threshold(self):
        model, trainer = self._make_trainer(sync_every=5)
        for i in range(1, 5):
            self.assertFalse(trainer.post_train_step(i))

    def test_sync_triggers_at_threshold(self):
        model, trainer = self._make_trainer(sync_every=3)
        trainer.post_train_step(1)
        trainer.post_train_step(2)
        self.assertTrue(trainer.post_train_step(3))

    def test_local_step_resets_after_sync(self):
        model, trainer = self._make_trainer(sync_every=3)
        for i in range(1, 4):
            trainer.post_train_step(i)
        self.assertEqual(trainer._local_step, 0)

    def test_sync_without_cross_replica_pg(self):
        model, trainer = self._make_trainer(sync_every=2)
        trainer.post_train_step(1)
        self.assertTrue(trainer.post_train_step(2))

    def test_sync_calls_allreduce_per_param(self):
        mock_work = MagicMock()
        mock_work.wait = MagicMock()
        mock_pg = MagicMock()

        with patch("torch.distributed.all_reduce", return_value=mock_work) as mock_ar:
            model, trainer = self._make_trainer(sync_every=2, cross_replica_pg=mock_pg)
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            trainer.post_train_step(1)
            trainer.post_train_step(2)

            num_params = len(list(model.parameters()))
            self.assertEqual(mock_ar.call_count, num_params)

    def test_state_dict_round_trip(self):
        model, trainer = self._make_trainer(sync_every=5)
        for i in range(1, 4):
            trainer.post_train_step(i)

        sd = trainer.state_dict()
        self.assertEqual(sd["local_step"], 3)
        self.assertIn("outer_optimizer", sd)

        model2, trainer2 = self._make_trainer(sync_every=5)
        trainer2.load_state_dict(sd)
        self.assertEqual(trainer2._local_step, 3)

    def test_multiple_sync_cycles(self):
        model, trainer = self._make_trainer(sync_every=2)
        synced = sum(1 for i in range(1, 9) if trainer.post_train_step(i))
        self.assertEqual(synced, 4)


# ---------------------------------------------------------------------------
# create_diloco_trainer factory
# ---------------------------------------------------------------------------

class TestCreateDiLoCoTrainer(unittest.TestCase):
    def test_returns_none_when_disabled(self):
        config = DiLoCoConfig(enabled=False)
        self.assertIsNone(create_diloco_trainer(config, [_simple_model()]))

    def test_returns_basic_trainer_without_lighthouse(self):
        config = DiLoCoConfig(enabled=True, lighthouse_addr="", pin_memory=False)
        result = create_diloco_trainer(config, [_simple_model()])
        self.assertIsInstance(result, DiLoCoTrainer)

    def test_returns_ft_trainer_with_lighthouse(self):
        """Skip if torchft not installed."""
        try:
            import torchft  # noqa: F401
        except ImportError:
            self.skipTest("torchft not installed")

        from megatron.core.distributed.diloco import FaultTolerantDiLoCoTrainer

        config = DiLoCoConfig(
            enabled=True,
            lighthouse_addr="localhost:29510",
            replica_id="test_replica",
            pin_memory=False,
        )
        mock_optimizer = MagicMock()
        # Patch at the torchft module level so the local imports inside
        # FaultTolerantDiLoCoTrainer.__init__ pick up the mocks.
        with patch("torchft.ProcessGroupGloo", MagicMock()):
            with patch("torchft.Manager", MagicMock()):
                result = create_diloco_trainer(
                    config, [_simple_model()], megatron_optimizer=mock_optimizer
                )
        self.assertIsInstance(result, FaultTolerantDiLoCoTrainer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
