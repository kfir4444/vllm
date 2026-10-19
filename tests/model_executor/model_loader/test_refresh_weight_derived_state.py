# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-derived state must be recomputable after weights are overwritten.

Elastic EP builds a new worker with dummy weights (post-processing included),
then overwrites the weights with real ones received from a peer. Tensors that
are derived from weights but are not in ``state_dict()`` are not transferred,
so they must be recomputed by ``refresh_weight_derived_state``.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm.model_executor.model_loader.utils import refresh_weight_derived_state


class _DerivedModel(nn.Module):
    """Mimics DeepSeek-V4's ``hc_attn_fn_broadcast``: a plain tensor attribute
    (not a buffer, so not in ``state_dict()``) summed from a weight."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(4, 3), requires_grad=False)
        self.derived: torch.Tensor | None = None
        self.hook_calls = 0
        self.process_weights_after_loading()

    def process_weights_after_loading(self) -> None:
        self.hook_calls += 1
        self.derived = self.w.detach().view(2, 2, 3).sum(dim=1)


def _fail_if_called(layer):
    raise AssertionError("quant methods must not run when refreshing derived state")


class _QuantProbe(nn.Module):
    """Stands in for a layer whose quant method must not be re-run."""

    def __init__(self):
        super().__init__()
        self.quant_method = SimpleNamespace(
            process_weights_after_loading=_fail_if_called
        )


def test_refresh_recomputes_derived_tensor_after_in_place_weight_update():
    model = _DerivedModel()
    assert "derived" not in model.state_dict()  # the bug: it is never transferred

    # "weight transfer": overwrite the dummy weights in place
    with torch.no_grad():
        model.w.copy_(torch.randn_like(model.w))
    stale = model.derived.clone()
    expected = model.w.detach().view(2, 2, 3).sum(dim=1)
    assert not torch.equal(stale, expected)

    refresh_weight_derived_state(
        model, SimpleNamespace(dtype=torch.float32), torch.device("cpu")
    )

    assert torch.equal(model.derived, expected)
    assert model.hook_calls == 2


def test_refresh_does_not_run_quant_methods():
    model = _DerivedModel()
    model.probe = _QuantProbe()
    refresh_weight_derived_state(
        model, SimpleNamespace(dtype=torch.float32), torch.device("cpu")
    )
    assert model.hook_calls == 2
