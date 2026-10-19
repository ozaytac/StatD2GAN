"""GPU-hour projection from smoke timings only (no metric is read) and the pre-declared ladder."""
import numpy as np


def run_seconds(epoch_seconds, total_epochs, warmup_epochs, eval_seconds):
    """Projected wall clock of one full run from a short probe. The probe's first epoch is a
    warmup epoch when warmup_epochs>0 (it also carries CUDA start-up cost: conservative), the
    rest are adversarial epochs at n_critic=3 (an upper bound: full runs use 2 after epoch 50)."""
    es = list(epoch_seconds)
    if warmup_epochs > 0:
        warm, adv = es[0], float(np.median(es[1:]))
    else:
        warm, adv = 0.0, float(np.median(es))
    return warmup_epochs * warm + (total_epochs - warmup_epochs) * adv + eval_seconds


def project(per_arm_seconds: dict, baseline_seconds: dict, n_runs_per_arm: int, safety: float,
            gpu_hours: float, ladder: list):
    """per_arm_seconds: arm -> seconds per run. Drops arms in `ladder` order until it fits."""
    active = dict(per_arm_seconds)
    dropped = []

    def hours(a):
        return safety * n_runs_per_arm * (sum(a.values()) + sum(baseline_seconds.values())) / 3600

    while hours(active) > gpu_hours and len(dropped) < len(ladder):
        dropped.append(ladder[len(dropped)]); active.pop(dropped[-1])
    return dict(active_arms=list(active), dropped=dropped, projected_hours=round(hours(active), 2),
                full_scope_hours=round(hours(per_arm_seconds), 2), fits=hours(active) <= gpu_hours,
                n_runs=n_runs_per_arm * (len(active) + len(baseline_seconds)))
