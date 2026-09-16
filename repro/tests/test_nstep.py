import numpy as np

from dqnselector.rainbow import NStepAccumulator, Transition


def tr(action, reward, done=False):
    s = np.zeros(4, dtype=bool)
    s[:action] = True
    ns = s.copy(); ns[action] = True
    return Transition(s, action, reward, ns, done)


def test_nstep_full_horizon():
    acc = NStepAccumulator(n_step=3, gamma=0.5)
    assert acc.push(tr(0, 1.0)) == []
    assert acc.push(tr(1, 2.0)) == []
    out = acc.push(tr(2, 4.0))
    assert len(out) == 1
    assert out[0].n_steps == 3
    assert abs(out[0].reward - (1.0 + 0.5 * 2.0 + 0.25 * 4.0)) < 1e-12


def test_nstep_terminal_flush_records_short_horizon():
    acc = NStepAccumulator(n_step=3, gamma=0.5)
    acc.push(tr(0, 1.0))
    out = acc.push(tr(1, 2.0, done=True))
    assert [x.n_steps for x in out] == [2, 1]
    assert out[0].done and out[1].done
    assert abs(out[0].reward - 2.0) < 1e-12
    assert abs(out[1].reward - 2.0) < 1e-12
