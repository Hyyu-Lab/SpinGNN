---
name: e2slp-evaluation
description: Evaluate e2SLP spin-lattice model uncertainty per structure for active-learning ranking and error correlation on labelled data. Accepts deployed TorchScript models or saved prediction arrays.
---

# e2SLP evaluation

Use the repository's [evaluate.py](../../tools/e2slp/evaluate.py)
for both uncertainty scoring and labelled correlation evaluation. Run commands
below from the SpinGNN repository root. For deployed inference, use the source
launcher and the Python environment installed for this checkout.

The e2SLP name follows the paper. "Per structure" describes the aggregation
level; **spin is included by default** through `sf_Uepi` and the local
spin-tangent projection. This is an evaluation tool for the spin-lattice model.
Existing model builders and array keys retain their code identifiers (`e2IP`,
`sf_Uepi`, `f_Uepi`) so trained checkpoints remain compatible.

## Definition and defaults

For spin-force covariance, use

`u(frame) = mean_i Tr(B_i^T sf_Uepi_i B_i)`.

`B_i` is an orthonormal 3-by-2 basis tangent to the spin direction. Spin norm
`<= 1e-6` means nonmagnetic and contributes zero. For direct inference this
threshold applies to the normalized `AtomicDataDict.SPIN_KEY` used by the model,
not the raw ASE magnetic moment. Thus a raw moment of `1e-8` is normalized
to a unit direction and remains magnetic. Offline NPZ arrays use the stored
spin vectors directly; their `1e-6` threshold is unchanged. The default mean includes
**all atoms**, matching the legacy notebook's `sf=True, max_mod=False`.
All-zero spins produce score zero with the default all-atom scope. Magnetic
scope requires at least one magnetic atom. Normalize directions for projection;
do not replace model input spin magnitudes or invent missing `spin_length`.

For `--target forces`, use the full Cartesian trace of `f_Uepi`.
`--reduction max` uses the maximum atom trace instead of a mean.
`--atom-scope magnetic` restricts both score aggregation and component RMSE
to nonzero-spin atoms. Default RMSE is the square root of the mean squared
error across **all selected atoms and three Cartesian components**.
The default does not project reference/predicted error vectors onto the tangent
plane, preserving the notebook convention. Only covariance is projected.

The score has squared target units. Keep physical Uepi without multiplying
training likelihood scales, residual scales or display multipliers. A fixed
positive display multiplier leaves Spearman unchanged, but changes thresholds.
Magnetic and all-atom means give the same ranking only when magnetic fractions
are constant. Label results clearly when using max or magnetic scope.

## Deployed model plus a trajectory

No labels are needed to score active-learning candidates:

```bash
validation/env/spingnn-cu128/bin/python tools/ci/source_python.py \
  tools/e2slp/evaluate.py \
  --model /path/to/deployed.pth --dataset /path/to/candidates.extxyz \
  --labels none --device cuda:0 --output /path/to/new_scores
```

For a labelled test set, use the same command with `--labels required`.
This requires reference labels for the selected target in every frame;
`--labels auto` (default) evaluates whichever labels are present, with the
number of usable frames reported separately for each error metric.
Force labels come from ASE calculator results; transverse spin-force labels
come from the `spin_forces_vert` array. Keep labels out of model inputs.

Chemical symbols are mapped to matching deployed type names by default.
For custom type names, pass e.g. `--species-map '{"Fe":"iron","O":"oxygen"}'`.
ASE-readable trajectories are supported; LAMMPS dumps must first have their
spin columns mapped to `initial_magmoms`. Preserve any real `spin_length`
field. If it is absent, AtomicData uses the input magnetic moment norm;
unit spin directions alone do not encode a separate magnetic magnitude.

Graph construction and inference use the deployed `default_dtype` metadata
(float32 or float64), including positions and cell before neighbor construction.
Integer indices keep their integer dtype. The caller's global precision settings
are restored before each yielded frame and on failure.

Force evaluation needs autograd even in eval mode. Do not wrap model inference
in `torch.no_grad()` or `torch.inference_mode()`. CPU is the default device;
select CUDA explicitly where available. `--limit N` supports a quick smoke run.

Use a fixed deployed checkpoint, recording its epoch externally if relevant.
This skill does not select epochs using test labels or start training/MD/DFT.

## Saved predictions without rerunning the model

```bash
validation/env/spingnn-cu128/bin/python tools/e2slp/evaluate.py \
  --predictions /path/to/predictions.npz --output /path/to/new_evaluation
```

Legacy fixed-atom NPZ schema:

| Field | Shape | Purpose |
|---|---|---|
| `sf_Uepi` (alias `Uepi`) | `(frames, atoms, 3, 3)` | spin-force covariance |
| `f_Uepi` (alias `Uepi`) | `(frames, atoms, 3, 3)` | force covariance with `--target forces` |
| `spin` | `(frames, atoms, 3)` | required for spin scoring or magnetic scope |
| `forces` (alias `force_hat`) | `(frames, atoms, 3)` | optional force prediction |
| `force_label` | `(frames, atoms, 3)` | optional force reference |
| `spin_forces` (alias `spin_fv_hat`) | `(frames, atoms, 3)` | optional transverse spin-force prediction |
| `spin_force_label` (alias `spin_fv_label`) | `(frames, atoms, 3)` | optional spin-force reference |
| `source`, `timestep` | `(frames,)` | optional provenance strings |

`--covariance-key KEY` overrides covariance lookup; ensure it is the physical
covariance for the selected target. Bare `.npy` covariance arrays need packaging
with correctly aligned spins as `.npz` before spin scoring.

New direct-inference archives flatten the atom dimension and store
`frame_ptr` (integer offsets, length frames+1). They can be reused with the same
`--predictions` command and support variable atom counts. Optional fields have
`<field>_present` masks; zero placeholders for absent labels are not references.
No pickle is needed. NaN/Inf values used for scoring or errors are rejected.

Direct inference spools per-atom predictions to temporary files in the output
directory, then writes compressed NPY members in fixed-size blocks. It retains
one frame of atom arrays plus frame metadata and scalar correlation rows, not
the full trajectory of atom arrays. Exact Spearman/Pearson still requires
structure-level rows proportional to the frame count. Ensure sufficient disk
space for the raw spool and compressed archive. Temporary files are removed on
success or failure; the archive is published only after writing completes.
Offline NPZ evaluation still loads its atom arrays into memory, but extracts
each provenance and presence-mask array only once.

## Outputs and interpretation

Output directory must be new or empty. Files:

- `per_structure.csv`: frame, source, timestep, atom/magnetic counts, physical
  uncertainty score and available structure component RMSEs.
- `report.json`: definition/options, frame counts, input/script SHA256,
  Spearman and Pearson for each available RMSE. Fewer than three labelled
  frames or constant score/error gives `null` and a reason, not a numerical rho.
- `correlation.png` / `.pdf`: labelled correlation plots; `--no-plot` disables.
- `predictions.npz`: produced by direct inference for repeatable offline scoring.

With default spin scoring, both force and spin-force correlations compare
**spin-force uncertainty** to each kind of error. Do not describe the force row
as using ordinary-force uncertainty. A single frame has a score, not a dataset
correlation. Structure-level ranking does not establish absolute calibration
or pooled atom-level correlation. Candidate scores do not require DFT labels;
actual uncertainty/error evaluation does.

For active learning, use CSV scores for ranking. This command does not itself
select or relabel structures. Preserve source/frame/timestep, deduplicate
geometries plus spins and apply a consistent unique-structure budget when
comparing selection strategies.
