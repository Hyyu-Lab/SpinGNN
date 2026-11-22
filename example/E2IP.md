# e2SLP training with Evidential Allegro

The [e2SLP method](https://doi.org/10.1103/PhysRevMaterials.10.103807) uses
tangent-plane evidential uncertainty to score magnetic configurations for
active learning. Builder names and YAML keys retain `e2IP` for checkpoint
compatibility.

Two builders extend the current energy models with equivariant evidence heads:

| Builder | Evidential target | Template |
| --- | --- | --- |
| `allegro.model.Allegro_e2IP` | `forces` | [06_allegro_e2ip.yaml](06_allegro_e2ip.yaml) |
| `allegro.model.Allegro_soc_e2IP` | `spin_forces_vert` or `forces` | [07_allegro_soc_e2ip.yaml](07_allegro_soc_e2ip.yaml) |

The first builder takes atomic structure as input. The SOC builder also
takes spins and includes exchange, anisotropy, biquadratic and SOC energy
terms. Atomic and spin forces remain energy derivatives. Use checkpoints
trained with the matching builder.

Edit the data/species/split paths in a template, then run from the repository:

```bash
python tools/ci/source_python.py nequip-train example/06_allegro_e2ip.yaml
python tools/ci/source_python.py nequip-train example/07_allegro_soc_e2ip.yaml
```

## Scaled training settings

Example 06 uses geometry-only training settings. Example 07 uses SOC
training settings for BiFeO₃. Replace the dataset and output paths, species,
and split sizes before training. Review the loss weights for your dataset.
`append: false` starts a fresh run.

| Setting | 06 geometry-only | 07 SOC |
| --- | --- | --- |
| Evidential target | `forces` | `spin_forces_vert` |
| `e2ip_likelihood_scale` | `1000.0` | `1000.0` |
| `e2ip_likelihood_dtype` | `float64` | `float64` |
| `e2ip_spin_force_scale` | `40.0` (unused) | `40.0` |
| `loss_reg` / `loss_epsilon` | `0.001` / `1.0e-6` | `0.001` / `1.0e-6` |
| Raw covariance trace metric | `total_f_Uepi_mean` | `total_sf_Uepi_mean` |
| Additional display metric | `total_f_Uepi_x1_mean` | `total_sf_Uepi_x100_mean` |

The SOC template uses two layers in each branch, latent dimensions
`[256, 256]`, and loss weights energy/force/stress/spin-force =
`20/10/10/0.01`.

`max_epochs: 10000000` is the upper cap. The configured learning-rate and
validation-loss stopping rules normally stop training earlier. Choose a
smaller cap for a short experiment.

Select `use_e2IP_loss: true` for the training objective, and
`validation_use_e2IP_loss: true` to validate with the same objective. Otherwise
validation uses the usual MSE loss. `evidential_key` selects exactly one target
and must appear in `loss_coeffs`. All other losses keep their usual meaning.
For SOC force evidence, set `evidential_key: forces`; the builder combines
geometry and SOC raw evidence with `soc_f_coeff` (default 0.01).

The evidential objective is a multivariate Student-t NLL plus an error-weighted
evidence regularizer:

```text
loss_coeffs[target] * (NLL + loss_reg * regularizer)
regularizer = (nu + kappa) * norm(residual)
```

Thus independent weights `a * NLL + b * regularizer` require the target weight
`a` and `loss_reg: b/a`. The legacy spin residual factor defaults to
`e2ip_spin_force_scale: 40.0`; both residual and covariance are transformed
consistently in the likelihood. Outputs are converted back to target units.
`loss_epsilon` is Cholesky jitter in the likelihood's working squared units.
Use explicit fixed loss weights; automatic/detached weighting is unsupported.

For small covariances, `e2ip_likelihood_scale: 1000.0` changes numerical
working units only: residuals multiply by 1000, covariance and jitter by
1e6, and the NLL subtracts `d * log(1000)`. The regularizer uses the original
residual. Thus loss values, gradients, and relative loss weights are preserved
up to rounding. This option defaults to 1 and is separate from the legacy
`e2ip_spin_force_scale`, which affects the regularizer and head units.
`e2ip_likelihood_dtype: float64` promotes residual subtraction, tangent
projection and the small-matrix likelihood calculations while keeping the
model and its parameters at their configured dtype. The default `float32`
setting retains the input dtype, including float64 models. Numerical scaling
alone does not improve relative precision or change covariance conditioning.

To log a larger uncertainty value alongside the physical value, use:

```yaml
metrics_components:
  - [total_sf_Uepi, mean, {functional: Trace}]
  - [total_sf_Uepi, mean, {functional: Trace, functional_kwargs: {scale: 100.0, units: x100}}]
```

The second column is named `total_sf_Uepi_x100_mean`; it multiplies only the
reported trace by 100, without changing model outputs, losses or physical
units. Geometry-only uses `total_f_Uepi` with scale `1.0` and suffix `x1`.
These multipliers do not guarantee a value near 0.1 as training evolves.
The training metrics above trace the per-structure **sum** of covariance
tensors; the evaluation script instead averages atomic traces. They also
average over different samples (batches versus a fixed deployed model), so
their numerical values should not be compared directly. Small Uepi values
are representable in float32; these options do not cure poorly calibrated
or overconfident uncertainty.

Keep `l_max >= 1`, `pad_to_alignment: 1` (the default), `spingnn_compile.enabled: false` and
`train_autocast.enabled: false`. The matrix exponential and derivative path
are validated in eager float32/float64. Compilation, mixed precision,
and distributed execution have not been validated. TorchScript deployment
and trajectory inference have been checked for the local geometry-only and
SOC checkpoints; LAMMPS integration of these new heads has not been validated.
The loss supplies additive numerator/count reductions
for the existing distributed trainer, but that alone is not a multi-GPU test.

## Outputs and Units

Let `p` be `f` for atomic forces or `sf` for tangent spin forces:

| Key | Shape | Meaning |
| --- | --- | --- |
| `p_nu`, `p_kappa` | `(N,1)` | Positive NIW evidence parameters |
| `p_sigma` | `(N,3,3)` | Cartesian NIW scale matrix in squared target units |
| `p_Uepi` | `(N,3,3)` | `nu * sigma / (kappa * (nu - d - 1))` |
| `total_p_Uepi` | `(B,3,3)` | Per-structure sum of epistemic tensors |
| `p_uncertainty` | `(B,1)` | Mean epistemic trace over valid atoms |

`d=3` for atomic forces and `d=2` for spin forces. Spin covariance is projected
with `P = I - s s^T`, using unit spin directions. Its radial eigenvalue is zero;
it is positive definite on the tangent plane. Atoms with spin norm at most
`1e-6` are masked before Cholesky and contribute neither to the spin loss nor
to the graph score denominator. An all-nonmagnetic structure has zero spin
score and loss. Covariances and scores scale by the square of global energy
rescaling; `nu` and `kappa` are dimensionless. Cartesian matrix fields have
`None` irreps metadata rather than an incorrect six-component vector irrep.

## Reproducible training check

```bash
python tools/ci/source_python.py tools/ci/e2ip_training_smoke.py \
  --output validation/results/e2ip-smoke-new --epochs 10 --device cuda:0
```

Use a new output directory. The script creates deterministic 16-frame Fe3O
datasets with derivative-consistent energy, force and stress labels; the SOC
dataset also has unit spins, tangent spin-force labels and a nonmagnetic site.
It runs the full training CLI for onlyr forces, SOC spin forces and SOC forces
(12 training / 4 validation structures, batch size 2), checks finite metrics,
loads each best checkpoint and verifies finite uncertainty predictions. It
also checks saved optimizer state and serialized evidential-loss flags.

The output directory contains the generated datasets, training configs, logs,
epoch metrics, checkpoints and `summary.json`.

## e2SLP evaluation and correlation

Use [`tools/e2slp/evaluate.py`](../tools/e2slp/evaluate.py) for
deployed-model trajectory scoring or saved-array analysis. The default SOC
score is the all-atom mean of the spin-tangent covariance trace, with zero
contribution from nonmagnetic atoms. Labelled inputs additionally produce
structure component RMSE and dataset Spearman/Pearson correlations. Unlabelled
inputs need no reference energy, forces, stress or spin forces.

See [`e2slp-evaluation`](../skills/e2slp-evaluation/SKILL.md) for
commands, input schema, output files and the exact masking/aggregation contract.

Direct trajectory evaluation uses the deployed precision before building the
neighbor graph and scores the normalized spin directions actually supplied to
the model. It streams per-atom outputs through temporary disk files into a
compatible flattened NPZ; only per-structure metadata and scalar statistics
grow with the number of frames in host memory. Offline NPZ scoring preserves
the stored-spin threshold and loads each provenance/presence field once.
