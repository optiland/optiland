"""Stopping criteria, observers and CancelToken (opt v2 PR5.1).

Kramer Harrison, 2026
"""

from __future__ import annotations

import pickle
import threading

import numpy as np
import pytest

from optiland.optimization import (
    CancelToken,
    CheckpointObserver,
    ConsoleObserver,
    CostTolerance,
    GradNormTolerance,
    HistoryObserver,
    MaxEvals,
    MaxIter,
    StepStall,
)
from optiland.optimization.native import dls
from optiland.optimization.observers import Observer
from optiland.optimization.state import OptimizationState
from optiland.optimization.stopping import StoppingBase, StoppingCriterion

from .validation.problems import PROBLEMS, CallableEvaluator

pytestmark = pytest.mark.filterwarnings("ignore")


def _evaluator():
    return CallableEvaluator(PROBLEMS["rosenbrock"].residuals, np.array([-1.2, 1.0]))


def _state(**kw):
    return OptimizationState(x=np.zeros(2), value=kw.pop("value", 1.0), **kw)


class _Fires(StoppingBase):
    """Fires on given iterations; counts calls and resets."""

    def __init__(self, *iterations):
        self.iterations = set(iterations)
        self.calls = 0
        self.resets = 0

    def reset(self, state):
        self.resets += 1

    def should_stop(self, state):
        self.calls += 1
        if state.iteration in self.iterations:
            return True, f"fired@{state.iteration}"
        return False, None


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------


def test_max_iter_and_max_evals():
    assert not MaxIter(3).should_stop(_state(iteration=2))[0]
    assert MaxIter(3).should_stop(_state(iteration=3))[0]
    assert not MaxEvals(10).should_stop(_state(n_evals=9))[0]
    assert MaxEvals(10).should_stop(_state(n_evals=10))[0]


def test_cost_tolerance_needs_a_step_and_relative_change():
    crit = CostTolerance(1e-3)
    crit.reset(_state(value=10.0))
    # No accepted step yet: a zero change is meaningless
    assert crit.should_stop(_state(value=10.0, iteration=0)) == (False, None)
    assert not crit.should_stop(_state(value=5.0, iteration=1))[0]
    stop, reason = crit.should_stop(_state(value=5.0 - 1e-6, iteration=2))
    assert stop
    assert "cost_tol" in reason
    # NaN never stops the run
    assert not CostTolerance().should_stop(_state(value=float("nan"), iteration=1))[0]


def test_grad_norm_and_step_stall_inactive_without_data():
    assert not GradNormTolerance(1.0).should_stop(_state())[0]
    assert not StepStall(1.0).should_stop(_state())[0]
    assert GradNormTolerance(1.0).should_stop(_state(grad_norm=0.5))[0]
    assert GradNormTolerance(1.0).should_stop(_state(gradient=np.array([0.1, 0.2])))[0]
    assert not GradNormTolerance(1e-3).should_stop(_state(grad_norm=0.5))[0]
    assert StepStall(1e-3).should_stop(_state(step_norm=1e-6))[0]
    assert not StepStall(1e-3).should_stop(_state(step_norm=1.0))[0]


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_or_stops_on_either_and_evaluates_both():
    a, b = _Fires(5), _Fires(2)
    crit = a | b
    assert isinstance(crit, StoppingCriterion)
    crit.reset(_state())
    assert (a.resets, b.resets) == (1, 1)
    assert crit.should_stop(_state(iteration=1)) == (False, None)
    assert crit.should_stop(_state(iteration=2)) == (True, "fired@2")
    assert (a.calls, b.calls) == (2, 2)


def test_and_latches_children():
    # MaxIter fires only once; a non-latching "and" would never stop.
    crit = _Fires(3) & _Fires(5)
    crit.reset(_state())
    assert not crit.should_stop(_state(iteration=3))[0]
    assert not crit.should_stop(_state(iteration=4))[0]
    stop, reason = crit.should_stop(_state(iteration=5))
    assert stop
    assert "fired@3" in reason
    assert "fired@5" in reason


def test_and_reset_clears_latches():
    crit = _Fires(1) & _Fires(1)
    crit.reset(_state())
    assert crit.should_stop(_state(iteration=1))[0]
    crit.reset(_state())
    assert not crit.should_stop(_state(iteration=0))[0]


def test_real_criteria_compose():
    crit = MaxIter(2) & CostTolerance(1e-3)
    crit.reset(_state(value=1.0))
    assert not crit.should_stop(_state(value=0.5, iteration=1))[0]
    # MaxIter fires here, cost still moving
    assert not crit.should_stop(_state(value=0.25, iteration=2))[0]
    # Cost converges later: both have now fired
    assert crit.should_stop(_state(value=0.25, iteration=3))[0]


# ---------------------------------------------------------------------------
# CancelToken
# ---------------------------------------------------------------------------


def test_cancel_token_flag_and_reuse():
    token = CancelToken()
    assert not token.cancelled
    token.cancel()
    assert token.cancelled
    token.reset()
    assert not token.cancelled


def test_cancel_token_from_another_thread_stops_dls():
    token = CancelToken()
    thread = threading.Thread(target=token.cancel)
    thread.start()
    thread.join()
    res = dls(_evaluator(), cancel_token=token)
    assert res.status == -4
    assert res.nit == 1
    assert "cancelled by user" in res.stop_reason
    assert not res.success


# ---------------------------------------------------------------------------
# Observers on native DLS
# ---------------------------------------------------------------------------


def test_observer_protocol_and_hooks_fire_in_order():
    class Recorder:
        def __init__(self):
            self.events = []

        def on_start(self, state):
            self.events.append(("start", state.iteration))

        def on_step(self, state):
            self.events.append(("step", state.iteration))

        def on_end(self, state, result):
            self.events.append(("end", state.iteration, result.nit))

    rec = Recorder()
    assert isinstance(rec, Observer)
    res = dls(_evaluator(), observers=[rec])
    kinds = [e[0] for e in rec.events]
    assert kinds[0] == "start"
    assert kinds[-1] == "end"
    assert kinds.count("step") == res.nit
    assert rec.events[-1] == ("end", res.nit, res.nit)
    assert [e[1] for e in rec.events if e[0] == "step"] == list(range(1, res.nit + 1))


def test_history_observer_attached_to_result():
    history = HistoryObserver()
    res = dls(_evaluator(), observers=[history])
    assert len(history.records) == res.nit + 1
    assert history.records[0]["iteration"] == 0
    assert history.records[-1]["value"] == pytest.approx(
        float(res.fun @ res.fun), rel=1e-12
    )
    assert res.history == history.records
    values = [r["value"] for r in history.records]
    assert all(b < a for a, b in zip(values, values[1:], strict=False))
    assert dls(_evaluator()).history is None


def test_stop_criterion_ends_dls_with_reason():
    res = dls(_evaluator(), stop=MaxIter(3), ftol=1e-14, xtol=1e-14, gtol=1e-14)
    assert res.status == -4
    assert res.nit == 3
    assert "max_iter=3" in res.message
    assert res.stop_reason == res.message
    assert not res.success


def test_stop_max_evals_counts_jacobian_columns():
    res = dls(_evaluator(), stop=MaxEvals(8), ftol=1e-14, xtol=1e-14, gtol=1e-14)
    assert res.status == -4
    assert res.n_evals == res.nfev


def test_observer_stop_iteration_is_clean_stop():
    class Stopper:
        def on_start(self, state):
            pass

        def on_step(self, state):
            if state.iteration == 2:
                raise StopIteration

        def on_end(self, state, result):
            pass

    res = dls(_evaluator(), observers=[Stopper()])
    assert res.status == -2
    assert res.nit == 2


def test_state_is_consistent_at_on_step():
    seen = []

    class Probe:
        def on_start(self, state):
            seen.append(("start", state.value, state.x.copy()))

        def on_step(self, state):
            r = PROBLEMS["rosenbrock"].residuals(state.x)
            seen.append(("step", state.value, r @ r, state.step_norm, state.n_evals))

        def on_end(self, state, result):
            seen.append(("end", state.stop_reason, result.message))

    dls(_evaluator(), observers=[Probe()])
    start = seen[0]
    r0 = PROBLEMS["rosenbrock"].residuals(start[2])
    assert start[1] == pytest.approx(r0 @ r0, rel=1e-12)
    for entry in seen[1:-1]:
        assert entry[1] == pytest.approx(entry[2], rel=1e-12)
        assert entry[3] > 0
    assert seen[-1][1] == seen[-1][2]
    evals = [e[4] for e in seen[1:-1]]
    assert evals == sorted(evals)


def test_callback_and_observers_both_fire():
    calls = []
    history = HistoryObserver()
    dls(_evaluator(), callback=calls.append, observers=[history])
    assert len(calls) == len(history.records) - 1


# ---------------------------------------------------------------------------
# Console and checkpoint
# ---------------------------------------------------------------------------


def test_console_observer_prints(capsys):
    res = dls(_evaluator(), observers=[ConsoleObserver(log_every=2, prefix="> ")])
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("> start:")
    assert out[-1].startswith("> end:")
    assert f"{res.nit} steps" in out[-1]
    assert len(out) == 2 + res.nit // 2


def test_checkpoint_observer_writes_and_prunes(tmp_path):
    ckpt = CheckpointObserver(tmp_path / "ck", every=2, keep_last=2)
    res = dls(_evaluator(), observers=[ckpt])
    files = sorted((tmp_path / "ck").glob("checkpoint_iter*.pkl"))
    assert len(files) == 2
    assert files[-1].name.endswith("_final.pkl")
    with files[-1].open("rb") as handle:
        state = pickle.load(handle)
    assert state.iteration == res.nit
    assert state.stop_reason == res.message


def test_checkpoint_keeps_all_by_default(tmp_path):
    res = dls(_evaluator(), observers=[CheckpointObserver(tmp_path, every=3)])
    expected = res.nit // 3 + 1
    assert len(list(tmp_path.glob("*.pkl"))) == expected


def test_checkpoint_write_failure_does_not_stop_run(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    ckpt = CheckpointObserver(tmp_path / "ok", every=1)
    ckpt._dir = blocker  # writing into a file path fails
    ckpt.on_start = lambda state: None
    res = dls(_evaluator(), observers=[ckpt])
    assert res.success
