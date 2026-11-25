# SpinGNN2 Performance Benchmarks

All native inference and LAMMPS numbers below use full-width NoSOC, SOC, and
DREAM models with prepared offline JIT artifacts and strict IEEE FP32 math
unless explicitly labeled TF32. See [VALIDATION.md](VALIDATION.md) for
numerical accuracy gates and test protocols.

## 1. Single-GPU Direct Inference (RTX 5090, 32 GB, sm120)

56 directed neighbors per center atom; 32,768 rows are matched ABBA hot medians
(µs/atom, strict IEEE FP32 `CUBLAS_COMPUTE_32F`), and 2,000,000 rows measure
the single-GPU capacity ceiling (`112,000,000` directed edges, `capacity`
profile). PyTorch eager OOMed at 32,768 atoms for all three models.

| Model | 32,768 Baseline (µs/atom) | 32,768 Optimized (µs/atom) | Speedup | 2,000,000 atoms (µs/atom) | Peak VRAM at 2,000,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Allegro_nosoc` (`mul=16`) | 4.48 | **3.77** | **1.19× (+18.7%)** | **4.63** | **1.58 GB** |
| `Allegro_soc` (`mul=16`) | 10.69 | **9.15** | **1.17× (+16.8%)** | **9.52** | **2.23 GB** |
| `Allegro_born` DREAM | 4.17 | **3.94** | **1.06× (+5.7%)** | **4.13** | **2.01 GB** |
| `Allegro_soc` prod (`mul=32`) | 52.98 | **42.33** | **1.25× (+25.2%)** | — | — |
| `Allegro_e2IP` | 5.54 | **5.00** | **1.11× (+10.8%)** | — | — |

On **RTX 4090 (`sm_89`)** at 32,768 atoms (matched ABBA strict IEEE FP32), the
same P1–P5 native optimizations improved `nosoc` by **+14.3%** (`235.36 → 205.87 ms`),
`soc` by **+15.8%** (`542.23 → 468.22 ms`), `born` by **+1.9%**, and
`soc-prod` (`mul=32`) by **+29.5%** (`2,708.96 → 2,091.73 ms`).

### LAMMPS 300 K thermal MD ABBA with bucketed CUDA Graphs (RTX 5090)

Bucketed tile padding (`align_up(E_tile + 128, 256)`) with device-side live-edge
masking achieves **100% post-warmup CUDA Graph replay (`fb=0`)** across thermal
cutoff crossings between neighbor rebuilds (strict IEEE FP32, matched node):

| Workload | Integrator | Baseline (ms/step) | Optimized (ms/step) | Speedup | Graph replays |
| --- | --- | ---: | ---: | ---: | --- |
| `nosoc` 4k | NVE | 22.35 | **19.01** | **+17.6%** | — |
| `born` 4k | NVE | 19.67 | **17.79** | **+10.6%** | — |
| `soc` 4k | NVE | 46.36 | **40.04** | **+15.8%** | — |
| `soc` 4k | Moving SIB | 63.45 | **55.62** | **+14.1%** | — |
| `soc` 4k | Moving CFI (`cuda_graph yes`) | 45.30 | **38.40** | **+18.0%** | 0 → 4/4 (100%) |
| `soc` 4k | Frozen CFI (`cuda_graph yes`) | 35.24 | **28.36** | **+24.3%** | 4/4 (100%) |
| `soc` 32k | NVE (`cuda_graph yes`) | 362.55 | **306.58** | **+18.3%** | 0 → 4/4 (100%) |
| `soc` 32k | Moving CFI (`cuda_graph yes`) | 342.33 | **304.96** | **+12.3%** | 4/4 (100%) |

### Stage-1 lattice cache in Python (4,096 atoms, RTX 5090)

| Model & math | Moving µs/atom (speedup) | Frozen µs/atom (speedup) | Device MiB retained (moving / frozen) |
| --- | ---: | ---: | ---: |
| NoSOC, strict FP32 | 3.585 (1.251×) | 2.470 (1.811×) | 1,754.89 / 119.89 |
| NoSOC, TF32 | 3.091 (1.251×) | 2.097 (1.848×) | 1,754.89 / 119.89 |
| SOC, strict FP32 | 8.415 (1.263×) | 4.524 (2.351×) | 3,742.39 / 247.64 |
| SOC, TF32 | 6.973 (1.265×) | 3.776 (2.341×) | 3,742.39 / 247.64 |

All eight cached configurations retained zero bytes on host.

## 2. Thermal LAMMPS MD with GPU Physical-Edge Compaction

### Single RTX 5090 (4,096 atoms, displacement-triggered rebuilds)

NVE: 192 timed steps at 1 fs. Moving SIB: 320 timed steps at 0.5 fs (300 K
spin bath, $\alpha = 0.05$). Zero dangerous builds across all runs.

| Model | Integrator | Skin (Å) | Steps/s | ns/day | Mean K | Timed rebuilds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| NoSOC | NVE | 0.4 | 43.43 | 3.753 | 323.2 | 8 |
| NoSOC | NVE | 1.5 | 43.77 | 3.781 | 323.2 | 2 |
| NoSOC | Moving SIB | 0.4 | 28.34 | 1.224 | 328.9 | 7 |
| NoSOC | Moving SIB | 1.5 | 28.40 | 1.227 | 328.9 | 1 |
| SOC | NVE | 0.4 | 18.36 | 1.586 | 722.9 | 22 |
| SOC | NVE | 1.5 | 18.34 | 1.585 | 722.9 | 5 |
| SOC | Moving SIB | 0.4 | 13.42 | 0.580 | 632.7 | 13 |
| SOC | Moving SIB | 1.5 | 13.42 | 0.580 | 632.7 | 3 |
| DREAM | NVE | 0.4 | 47.55 | 4.109 | 1651.3 | 30 |
| DREAM | NVE | 1.5 | 47.89 | 4.138 | 1663.0 | 7 |

GPU physical-edge compaction keeps throughput within 1% between 0.4 Å and
1.5 Å neighbor skins across all three models. On a steady 4,096-atom NVE
list (56 physical neighbors), compaction improved per-step latency to
**5.21 µs/atom** (NoSOC), **10.92 µs/atom** (SOC), and **4.48 µs/atom**
(DREAM) at both skins (up to 2.23× speedup at 1.5 Å skin).

### Single RTX 4090 (sm89, 4,096-atom NoSOC NVE, 1 fs)

| Skin (Å) | Steps/s | ns/day | Timed rebuilds | Energy range (eV/atom) |
| ---: | ---: | ---: | ---: | ---: |
| 0.4 | 33.97 | 2.935 | 8 | 6.2e-7 |
| 1.5 | 34.03 | 2.940 | 1 | 6.2e-7 |

## 3. Multi-GPU Weak Scaling (RTX 5090)

### 2.05 Million Atoms (512,000 atoms/GPU, 10 NVE steps at 0.02 fs)

| RTX 5090 GPUs | Total atoms | Seconds / 10 steps | Weak efficiency | Energy range (eV/atom) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 512,000 | 33.0036 | 100.00% | 1.93e-07 |
| 2 | 1,024,000 | 33.1692 | 99.50% | 2.11e-07 |
| 4 | 2,048,000 | 33.2721 | 99.19% | 1.74e-07 |

### 300 K Thermal NoSOC Weak Scaling (8,000 atoms/GPU, 0.4 Å skin, 2 nodes × 2 GPUs)

| Integrator | GPUs | Nodes | Steps/s | ns/day | Timed rebuilds | Atoms migrating ranks / nodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NVE (1 fs) | 1 | 1 | 22.57 | 1.950 | 8 | 0 / 0 |
| NVE (1 fs) | 2 | 1 | 22.32 | 1.929 | 8 | 88 / 0 |
| NVE (1 fs) | 4 | 2 | 22.17 | 1.915 | 8 | 176 / 88 |
| Moving SIB (0.5 fs) | 1 | 1 | 14.50 | 0.626 | 7 | 0 / 0 |
| Moving SIB (0.5 fs) | 2 | 1 | 14.34 | 0.620 | 7 | 92 / 0 |
| Moving SIB (0.5 fs) | 4 | 2 | 14.32 | 0.618 | 7 | 180 / 90 |

## 4. CUDA Graph Replay & Kernel Profiling

Opt-in `cuda_graph yes` in `pair_style spingnn2/kk` provides 2.6%–4.5%
additional loop speedup on steady neighbor lists (512 to 8,000 atoms) with
40–64 MiB memory overhead and zero fallbacks. For frequent neighbor rebuilds
(every 5 steps), default eager launches (`cuda_graph no`) are faster because
recapture overhead outweighs launch savings.

Nsight Systems per-kernel breakdown data for 32,768-atom strict FP32 direct
inference is archived in [data/baseline_kernels.csv](data/baseline_kernels.csv).
Reproducible benchmark drivers live under [`benchmarks/`](../benchmarks/).

## 5. SpinGNN2 v2 `torch.compile` + `PhysMuon` Training vs SpinGNN++ v1

Matched A/B training benchmarks comparing SpinGNN++ v1 (Adam, `e3nn` / `T-e3nn`
eager) against SpinGNN2 v2 (`spingnn2.symmetry` + `spingnn2.v2`, zero `e3nn`
runtime import, edge `bfloat16` AMP, selective edge activation checkpointing,
fused `SparseCGContract`/`bipolar_spin_pair_j1_j2`, and whole-step CUDA-Graphed
5-step quintic `PhysMuon(cuda_graph=True)`) on **NVIDIA RTX 5090 (CUDA 12.8)**
and **RTX 4090** via [`benchmarks/bench_v2_train.py`](../benchmarks/bench_v2_train.py).
Dynamic (`dynamic=True`, 0 recompiles across shapes) and opt-in capped-bucket
(`<= 4` buckets, `mode="reduce-overhead"`) are reported separately with full
CUDA Graph pool reserved VRAM:

### RTX 5090 & RTX 4090 Summary (512 atoms / 14,958 directed edges)

| Family (GPU) | v1 step (ms) | v2 `dynamic=True` (ms) | Dynamic speedup | v2 capped-bucket (ms) | Bucket speedup | VRAM reduction (alloc / reserved) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nosoc` (5090) | — | 1.36 ms opt | **3.29×** (`2.39×–3.29×` 64→512) | — | **7.91×** (`7.02×–7.91×`) | **-91.4% alloc / -85.1% res** |
| `soc` (5090) | — | 1.62 ms opt | **3.42×** (`2.91×–3.42×` 64→512) | — | **9.06×** (`8.99×–9.44×`) | **-85.8% alloc / -70.9% res** |
| `born` (5090) | — | 0.88 ms opt | **3.80×** (`2.88×–3.80×` 64→512) | — | **8.15×** (`7.32×–8.15×`) | **-88.4% alloc / -75.4% res** |
| `e2slp` (4090) | 102.70 | **30.91** (`28.1+2.79`) | **3.32×** (`3.32×–3.66×` 64→512) | **11.17** (`8.7+2.49`) | **9.20×** (`9.20×–9.55×`) | **-88.6% alloc / -80.8% res** |
| `nosoc` / `soc` / `born` (4090) | — | — | **2.76× / 3.16× / 3.96×** | — | **7.65× / 11.45× / 10.26×** | **>80%–90%** |
