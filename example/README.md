# Training examples

Choose one of the seven configurations and follow the
[training tutorial](TUTORIAL_TRAINING.md).

| File | Model | Intended use |
| --- | --- | --- |
| [01_structure_only.yaml](01_structure_only.yaml) | `allegro.model.Allegro` | Energy, atomic-force, and stress training without spins |
| [02_spin_nosoc.yaml](02_spin_nosoc.yaml) | `allegro.model.Allegro_nosoc` | Magnetic training without spin-orbit coupling |
| [03_spin_soc.yaml](03_spin_soc.yaml) | `allegro.model.Allegro_soc` | Magnetic training with spin-orbit coupling |
| [04_born_effective_charge.yaml](04_born_effective_charge.yaml) | `allegro.model.Allegro_born` | Energy, force, stress, and Born effective-charge training |
| [05_spin_soc_production.yaml](05_spin_soc_production.yaml) | `allegro.model.Allegro_soc` | Larger SOC model with full metrics, EMA, scheduling, and early stopping |
| [06_allegro_e2ip.yaml](06_allegro_e2ip.yaml) | `allegro.model.Allegro_e2IP` | Geometry-only force evidence |
| [07_allegro_soc_e2ip.yaml](07_allegro_soc_e2ip.yaml) | `allegro.model.Allegro_soc_e2IP` | SOC tangent spin-force or atomic-force evidence |

See [e2SLP training and outputs](E2IP.md) for loss settings, covariance units,
and a reproducible GPU training check. For response-property fine-tuning,
start with the [packaged BEC workflow](#born-effective-charges-on-a-packaged-foundation-model)
or the [dielectric and piezoelectric heads](#dielectric-and-piezoelectric-response-heads-on-a-packaged-foundation-model).

The NoSOC example intentionally uses the dedicated `Allegro_nosoc` builder.
Do not replace it with `Allegro_soc` plus `lsoc_layer: false`: that setting only
disables the additional SOC layer, while the SOC builder can still contain a
full exchange matrix and magnetic-anisotropy terms. `Allegro_nosoc` instead
uses interactions invariant under rotations of the spins independently of the
atomic structure.

## Before training

Edit the selected YAML file and set at least:

- `dataset_file_name`
- `chemical_symbol_to_type`
- `n_train` and `n_val`
- `r_max`
- `loss_coeffs` and `metrics_components` to match the labels in the dataset

For the Born effective-charge example, also update `born_scale_list`. It must
contain one value per chemical type in the type-index order specified by
`chemical_symbol_to_type`.

The example paths assume that commands are run from the repository root. The
dataset is not included in this repository.

Run the examples through the [source launcher](../docs/SOURCE_CHECKOUT.md)
to select both packages from the checkout.

## extxyz fields

All examples expect an ASE-readable extxyz file. The default loss sections
expect the following labels:

- Structure-only: `energy`, `forces`, and `stress`.
- Magnetic: the same labels plus the per-atom `initial_magmoms` input and
  `spin_forces_vert` target.
- Born effective charge: `energy`, `forces`, `stress`, and the per-atom
  `born_effective_charges` target. Each Born effective-charge tensor is stored
  as nine flattened components. The system-level
  `total_born_effective_charges` zero target is generated internally to impose
  the acoustic sum rule.

Each `initial_magmoms` or `spin_forces_vert` value is a three-component vector
per atom. If the dataset has no stress or spin-force labels, remove that key
from both `loss_coeffs` and `metrics_components`. A magnetic model still needs
`initial_magmoms`, even when spin-force supervision is disabled.

`spin_forces` is also a supported training target. If that is the label stored
in the dataset, replace `spin_forces_vert` with `spin_forces` in the selected
magnetic configuration.

The BEC model produces energy, forces, stress, per-atom
`born_effective_charges`, and system-level `total_born_effective_charges`.
For piezoelectric prediction in Python, use the
[packaged piezoelectric head](#dielectric-and-piezoelectric-response-heads-on-a-packaged-foundation-model).
To construct a BEC extxyz dataset from indexed VASP outputs, use:

```bash
python tools/ci/source_python.py nequip-data-mlff-bec \
  --dataset-path path/to/data.extxyz \
  --folder path/to/vasp-results \
  --out-path example/data/born_effective_charge.extxyz
```

The VASP-results directory is expected to contain matching `OUTCAR.<index>`
and `log.out.<index>` files.

## Run

Follow the [installation guide](../INSTALL.md), then choose one configuration:

```bash
python tools/ci/source_python.py nequip-train example/01_structure_only.yaml
python tools/ci/source_python.py nequip-train example/02_spin_nosoc.yaml
python tools/ci/source_python.py nequip-train example/03_spin_soc.yaml
python tools/ci/source_python.py nequip-train example/04_born_effective_charge.yaml
python tools/ci/source_python.py nequip-train example/05_spin_soc_production.yaml
python tools/ci/source_python.py nequip-train example/06_allegro_e2ip.yaml
python tools/ci/source_python.py nequip-train example/07_allegro_soc_e2ip.yaml
```

Run only one command at a time unless separate output directories and GPU
resources have been configured.

To parse and run a forward pass with synthetic inputs for all seven
configurations, run:

```bash
python tools/ci/source_python.py example/validate_configs.py
```

## Born effective charges on a packaged foundation model

`PackagedBackboneModel` runs the leading stages of a NequIP-family model
shipped as a `.nequip.zip` archive from NequIP 0.7 or newer. It exposes selected
intermediate tensors for response heads.
`allegro.model.TensorFeatureBEC` turns them into per-atom Born effective
charges. Add these keys to a configuration with the usual dataset and training
settings. The stage, module and field names are examples for a packaged
Allegro model; they depend on the package.

```yaml
model_builders:
  - PackagedBackboneModel
  - allegro.model.TensorFeatureBEC
chemical_symbols: [H, C, O]        # must give the package's type names, in order
r_max: 6.0                         # must equal the package's r_max
default_dtype: float64             # data, targets and losses
model_dtype: float32               # the packaged backbone keeps its own dtypes
spingnn_compile: {enabled: false}  # no force output to compile

packaged_backbone_path: path/to/foundation_model.nequip.zip
packaged_backbone_stop_after: allegro            # last packaged stage to run
packaged_backbone_taps:
  bec_tensor_features: {module: allegro.tps.2}   # output of a packaged submodule
  bec_scalar_features: edge_features             # packaged data keys
  bec_tensor_basis: edge_attrs
packaged_backbone_trainable_prefixes: [latents., tps.]  # all others are frozen

bec_readout_tensor_field: bec_tensor_features
bec_readout_scalar_field: bec_scalar_features
bec_readout_tensor_basis_field: bec_tensor_basis
bec_readout_num_tensor_features: 64              # channels of the tensor tap
bec_readout_hidden_channels: 64
bec_readout_antisymmetric_channels: 16
bec_readout_asr: hard                            # exact acoustic sum rule

dataset_include_keys: [use_bec]                  # integer extxyz info key, 0 or 1
loss_coeffs:
  born_effective_charges: [1.0, PerFrameL1Loss, {mask_field: use_bec}]

optimizer_name: Adam
learning_rate: 1.0e-4
optimizer_param_groups:
  - {name: readout}                              # no match: all other parameters
  - {name: backbone, match: [packaged_backbone.], lr: 1.0e-5}
early_stopping_lr_group: readout                 # the group reported as LR
start_of_epoch_callbacks:
  - my_package.callbacks.on_epoch_start          # called as f(trainer)
```

- The SpinGNN type names must equal the package's type names in the same
  order, including elements absent from the data. `chemical_symbols` is sorted
  by atomic number; use `chemical_symbol_to_type` for any other order. Type
  names and `r_max` are checked when the model is built.
- A trainable prefix matches a packaged parameter name with or without its
  stage name, so `latents.` selects `allegro.latents.*`.
- `PerFrameL1Loss` averages over each frame's elements and weights frames
  equally. Frames with `use_bec` 0 add zero but still count as frames. A
  coefficient equal to the global number of frames per optimizer step
  (`world_size * batch_size`) applies the sum of the per-frame gradients.
- `optimizer_param_groups` patterns are name prefixes or `re:<regex>`, matched
  against the model's `state_dict` keys (here
  `packaged_backbone.func.<stage>.<rest>` and `bec_readout.<rest>`); the first
  match wins. Every trainable parameter must land in a group. Each group's
  learning rate is logged as `LR_<name>`.
- Start-of-epoch callbacks run on every rank. `trainer.iepoch` is the 0-based
  epoch about to run, also after a restart. Under DDP `trainer.model` is the
  `DistributedDataParallel` wrapper; `getattr(trainer.model, "module",
  trainer.model)` reaches the model in every mode.
- `packaged_backbone_torch_compile: true` compiles the packaged stages with
  `torch.compile` (dynamic shapes; checkpoints stay compatible with eager runs).
- The packaged stages always run in eval mode. The response head itself runs
  in eager mode. Evaluate this workflow with `nequip-evaluate --train-dir`.
  The packaged backbone and response head are outside the
  `nequip-deploy` interface.

## Dielectric and piezoelectric response heads on a packaged foundation model

Use `allegro.model.TensorFeatureDielectric` or `allegro.model.TensorFeaturePiezo`
after `PackagedBackboneModel`. The backbone must expose a strided tensor tap
with per-channel irreps supporting every requested tensor-product output;
`0e+1o+2e+3o` supports both heads. Keep the package path, cutoff, type names
and trainable prefixes configured as in the preceding example. For example:

```yaml
model_builders: [PackagedBackboneModel, allegro.model.TensorFeatureDielectric]
packaged_backbone_taps:
  tensor_features: {module: allegro.tps.2}
  scalar_features: edge_features
  tensor_basis: edge_attrs
TensorFeatureCartesianReadout_tensor_field: tensor_features
TensorFeatureCartesianReadout_scalar_field: scalar_features
TensorFeatureCartesianReadout_tensor_basis_field: tensor_basis
TensorFeatureCartesianReadout_num_tensor_features: 64
TensorFeatureCartesianReadout_hidden_channels: 64
TensorFeatureCartesianReadout_gated_convolution: true
TensorFeatureCartesianReadout_reduce: mean
ipv_readout_antisymmetric_channels: 16
ipv_readout_per_species_scale: 1.0  # choose a scale from training labels
ipv_readout_per_species_scale_trainable: true
loss_coeffs:
  ipv: [1.0, PerFrameL1Loss]
  epsilon_ion: [1.0, PerFrameL1Loss]
  epsilon_el: [1.0, PerFrameL1Loss]
```

Set `TensorFeatureCartesianReadout_memory_efficient_contraction: true` to reduce
activation memory for large graphs; results equal the default path up to
floating-point roundoff.

The shared class prefix applies to both readouts. Layer prefixes
`ipv_readout_*` / `eps_el_readout_*` or `ipd_readout_*` /
`piezo_el_readout_*` override hyperparameters. Each layer also accepts
`<layer>_construct_dtype` to initialize under that dtype before casting to
the build dtype. The dielectric frame predictor accepts
`system_eps_el_predict_mlp_latent_dimensions` (default `[64, 64]`),
`system_eps_el_predict_mlp_nonlinearity` (`silu`) and
`system_eps_el_predict_tensor_hidden_channels` (8). Use the analogous
`system_piezo_el_predict_` prefix for piezoelectric predictor settings. Bare
`mlp_*` keys apply to every layer that accepts them; use these layer prefixes
to configure each predictor independently.

| Head | Atomic label | Frame labels | Units |
| --- | --- | --- | --- |
| Dielectric | `ipv` (N,9) | `epsilon_ion`, `epsilon_el` (B,9) | ipv: Angstrom³; epsilon: dimensionless |
| Piezoelectric | `ipd` (N,27) | `piezo_ion`, `piezo_el` (B,27) | ipd: (C/m²)·Angstrom³; piezo: C/m² |

For piezoelectric training select `TensorFeaturePiezo`, use the three piezo
loss keys and `ipd_readout_per_species_scale`; omit the ipv antisymmetric
setting. Both piezo outputs obey `T_ijk=T_ikj`; electronic dielectric output
obeys `X_ij=X_ji`. Atomic ipv is an unrestricted rank-2 tensor when
`ipv_readout_antisymmetric_channels` is set; otherwise the same-edge lift has
no antisymmetric (1e) part, so ipv is symmetric and the builder logs a warning.

Ionic readouts sum atomic responses into `total_ipv` / `total_ipd`, then
`VolumeNormalization` divides by `abs(det(cell))` without further constants.
Cells must have finite volume greater than 1e-12 in the supplied length units
cubed. Electronic readouts average atomic features before a frame predictor.
All heads use `asr: none`. Formulas, fields, frame reductions and electronic
antisymmetric settings are fixed by the builders and cannot be overridden.
The `system_eps_el_features`, `system_piezo_el_features`, `edge_ipd_contribution`
and `edge_ipv_antisymmetric` fields are model-internal intermediates, never
labels loaded from ASE; the system features are already shaped (B,9)/(B,27).

ASE input tensors may be flat or Cartesian arrays; they are flattened in
C order. An unbatched frame label is (9,) or (27,). These keys are registered
by default and require no `dataset_include_keys`. They are distinct from the
legacy `piezoelectric` key, which retains its (3,3,3) shape and e·Angstrom units.
Readouts and predictors run in `model_dtype`, while the packaged backbone
keeps its own dtype. Train all three losses to keep both heads connected to
backpropagation. Final readout weights start at zero; the bias-free dielectric
MLP can therefore give the electronic branch a slow start.

Response readouts and frame predictors are eager-only and cannot be deployed
with `nequip-deploy`. Packaged stages may use their separate opt-in compile
setting; this does not compile the response heads. Evaluate training runs
with `nequip-evaluate --train-dir`.
