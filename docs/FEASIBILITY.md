# Feasibility note (planning only; measured values come from `make smoke`)

Scope: 5 locations × (9 arms + 2 baselines) × 5 seeds = 275 runs; pilot = 30 of them (Ankara, Dubai × 3 arms × 5 seeds), kept.

Planning reference. Only `train_seconds` of the earlier v2 campaign was used, as a wall-clock prior, never a metric:
median 650 s (full model), 247 to 739 s across arms, 225 arm runs ≈ 34 GPU-h; earlier baselines ≈ 65 s each (≈ 0.9 GPU-h).
GPU model of that campaign is not recorded in the files. Pilot ≈ 5.4 GPU-h on the same basis.

Caveat: the new baselines are re-implemented (the earlier baseline code was not recoverable) with larger training budgets
(`protocol.yaml: baselines`), so their cost is unknown until smoke. Evaluation adds CPU time per run (metrics, isotonic maps), measured by smoke.

Decision rule if the projection exceeds `budget.gpu_hours`: drop arms in the pre-declared order (no_aux_losses, fixed_lambda,
moment_D_stat), never seeds or locations; decided from timings before any metric is read. If it still does not fit, the run stops for a human scope decision.

Not exercised in the authoring environment (no GPU, no PyPI/CDS access): torch code paths, RCGAN/TimeGAN, `download_era5.py`, real statsmodels calls.
Exercised: data split/embargo on the real ERA5 files, all numpy/scipy metrics, calibration, projection, Holm/Wilcoxon logic (against a reference implementation), bootstrap,
budget/ladder/gate/decision logic, and the full collect → analyze → figures → tables chain on synthetic run records.
