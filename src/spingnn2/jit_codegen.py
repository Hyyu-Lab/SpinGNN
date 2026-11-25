"""Emit model-specialized contraction kernels from the immutable M1 tables."""

from dataclasses import dataclass, field
import re
import struct


def _launch_bounds(gpu_arch, *, heavy=False):
    if not gpu_arch:
        return "128, 4"
    # sm_80/sm_90 have 2048 max threads/SM; sm_86/sm_89/sm_120 have 1536 max threads/SM.
    # For heavy TP reverse kernels, minBlocks=5 (96 regs/thread) keeps occupancy high with 0 stack spill.
    if heavy:
        return "128, 5" if gpu_arch in ("sm_80", "sm_90", "sm_120") else "128, 4"
    return "128, 6" if gpu_arch in ("sm_80", "sm_90") else "128, 4"


def _cuda_param(type_, name, *, restrict=True):
    if restrict and type_.endswith("*"):
        return f"{type_} __restrict__ {name}"
    return f"{type_} {name}"


@dataclass
class Source:
    backend: str
    precision: str
    gpu_arch: str = ""
    parts: list = field(default_factory=list)
    symbols: list = field(default_factory=list)

    @property
    def cuda(self):
        return self.backend == "cuda"

    def __post_init__(self):
        real = "float" if self.precision == "float32" else "double"
        self.parts.append(
            ("" if self.cuda else "#include <math.h>\n")
            + f"using Real = {real};\n"
            + (
                "#define SP2_INLINE __device__ __forceinline__\n"
                if self.cuda
                else "#define SP2_INLINE static inline\n"
            )
            + """SP2_INLINE Real sp2_ipow_pos(Real base, int exp) {
    Real res = Real(1);
    while (exp > 0) {
        if (exp & 1) res *= base;
        base *= base;
        exp >>= 1;
    }
    return res;
}"""
        )

    def real(self, value):
        # Hexadecimal literals preserve exactly the value encoded in the blob.
        number = float(value)
        if self.precision == "float32":
            number = struct.unpack("<f", struct.pack("<f", number))[0]
            return number.hex() + "f"
        return number.hex()

    def kernel(
        self,
        name,
        arguments,
        body,
        *,
        grid_body=None,
        shared=0,
        dynamic_shared=False,
        warp_rows=False,
        raw_cuda_body=None,
        heavy=False,
        no_restrict=(),
    ):
        self.symbols.append(name)
        if self.cuda:
            formal = ", ".join(
                _cuda_param(type_, arg, restrict=(arg not in no_restrict))
                for type_, arg in arguments
            )
            lb = _launch_bounds(self.gpu_arch, heavy=heavy)
            if raw_cuda_body is not None:
                cuda_body = raw_cuda_body
            elif grid_body is not None:
                row = (
                    "const int e = int(blockIdx.x * 4 + threadIdx.x / 32);\n"
                    "if (e >= n) return;\n"
                    if warp_rows
                    else "const int e = int(blockIdx.x);\n"
                )
                smem = (
                    "extern __shared__ Real scratch[];\n"
                    if dynamic_shared
                    else (f"__shared__ Real scratch[{shared}];\n" if shared else "")
                )
                cuda_body = row + smem + grid_body
            else:
                cuda_body = (
                    "const int idx = int(blockIdx.x * blockDim.x + threadIdx.x);\n"
                    "if (idx >= n) return;\n" + body
                )
            self.parts.append(
                f'extern "C" __global__ __launch_bounds__({lb}) void {name}({formal}) {{\n{cuda_body}\n}}'
            )
            return
        loads = "\n".join(
            f"const auto {arg} = *static_cast<{type_}*>(args[{i}]);"
            for i, (type_, arg) in enumerate(arguments)
        )
        self.parts.append(
            f'extern "C" void {name}(int, int, void** args) {{\n{loads}\n'
            "#pragma omp parallel for schedule(static) if(n >= 64)\n"
            f"for (int {'e' if grid_body else 'idx'} = 0; "
            f"{'e' if grid_body else 'idx'} < n; ++{'e' if grid_body else 'idx'}) {{\n"
            + (f"Real scratch[{shared}];\n" if shared else "")
            + body
            + "\n}\n}"
        )


def _name(name):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
        raise ValueError(f"Invalid JIT layer name {name!r}")
    return "sp2_trunk_blas_" + name


def _paths(tp):
    offsets = [tp["offsets_" + name] for name in ("in1", "in2", "out")]
    return [
        tuple(int(offsets[axis][row[axis]] + row[axis + 3]) for axis in range(3))
        + (value,)
        for row, value in zip(tp["w3j_indices"], tp["scaled_w3j"])
    ]


def _contractions(src, name, paths, d1, d2, d3, scalars):
    fwd = [[] for _ in range(d3)]
    lhs = [[] for _ in range(d1)]
    rhs = [[] for _ in range(d2)]
    for i, j, k, coeff in paths:
        w = src.real(coeff)
        fwd[k].append(f"sum += (f[{i}] * v[{j}]) * {w};")
        lhs[i].append((k, f"sum += (dy[{k}] * {w}) * v[{j}];"))
        rhs[j].append((k, f"sum += (dy[{k}] * {w}) * f[{i}];"))

    def emit(direction, entries, argument, limit=None):
        cases = []
        for k, group in enumerate(entries):
            terms = (
                group if limit is None else [expr for out, expr in group if out < limit]
            )
            if terms:
                cases.append(f"case {k}: " + " ".join(terms) + " break;")
        src.parts.append(
            f"SP2_INLINE Real {name}_{direction}(int k, {argument}, Real sum = 0) {{\n"
            + "switch(k) {\n"
            + "\n".join(cases)
            + "\n}\nreturn sum;\n}"
        )

    emit("product", fwd, "const Real* f, const Real* v")
    emit("left_scalar", lhs, "const Real* dy, const Real* v", scalars)
    emit("right_scalar", rhs, "const Real* dy, const Real* f", scalars)


def _packed_linear(src, name, spec, d3, mul):
    groups = []
    if spec is not None:
        if spec["dim_in"] != d3 or spec["mul_in"] != mul:
            raise ValueError("Invalid JIT Linear input dimensions")
        ni, no = spec["dim_in"], spec["dim_out"]
        mo = spec["mul_out"]
        packed_in = packed_out = 0
        for block in spec["blocks"]:
            dim, out = block["components"], block["out"]
            if not 0 <= out < out + dim <= no:
                raise ValueError("Invalid JIT Linear output block")
            for group in block["groups"]:
                start, stop = group["start"], group["stop"]
                if not 0 <= start < stop <= ni or (stop - start) % dim:
                    raise ValueError("Invalid JIT Linear input block")
                count = (stop - start) // dim
                if group["weight"].size != mul * count * mo:
                    raise ValueError("Invalid JIT Linear weight array")
                groups.append((dim, out, start, stop, count, packed_in, packed_out))
                packed_in += dim * count * mul
                packed_out += dim * mo
    else:
        no, mo = d3, mul

    if spec is None:
        return groups

    seed = []
    unpack = []
    for dim, out, start, stop, count, pi, po in groups:
        seed.append(
            f"if (q >= {po} && q < {po + dim * mo}) {{\n"
            f"const int local = q - {po}, j = local / {mo}, u = local % {mo};\n"
            f"packed[(long long)rows * {po} + e * {dim * mo} + local] = "
            f"value[e * {mo * no} + u * {no} + {out} + j];\n}}"
        )
        unpack.append(
            f"if (k >= {out} && k < {out + dim}) "
            f"sum += packed[(long long)rows * {po} + (e * {dim} + k - {out}) * {mo} + u];"
        )
    args = (
        ("int", "n"),
        ("int", "rows"),
        ("const Real*", "input"),
        ("Real*", "output"),
    )
    src.kernel(
        name + "_linear_seed",
        args,
        f"const long long e = idx / {max(1, packed_out)};\n"
        f"const int q = idx % {max(1, packed_out)};\n"
        "const Real* value = input; Real* packed = output;\n" + "\n".join(seed),
    )
    src.kernel(
        name + "_linear_output",
        args,
        f"const long long e = idx / {no * mo};\n"
        f"const int u = idx % {no * mo} / {no}, k = idx % {no};\n"
        "const Real* packed = input; Real sum = 0;\n"
        + "\n".join(unpack)
        + "\noutput[idx] = sum;",
    )
    return groups


def _tensor_layer(src, name, layer):
    tp = layer["tensor_product"]
    d1, d2, d3 = (tp[k] for k in ("dim_in1", "dim_in2", "dim_out"))
    mul, mul1 = tp["mul_out"], tp["mul_in1"]
    ns = layer["n_scalar_outs"]
    if min(d1, d2, d3, mul) <= 0 or mul1 not in (1, mul) or not 0 <= ns <= d3:
        raise ValueError("Invalid JIT tensor product dimensions")
    fw, ew, cw = d1 * mul1, d2 * mul, d3 * mul
    elem_bytes = 4 if src.precision == "float32" else 8
    env_bytes = ew * elem_bytes
    # Portable smem budget across sm_80/86/89/90/120 (min opt-in is 99 KiB on sm_86/89/120):
    # <=48 KiB uses static smem; (48, 96] KiB uses dynamic opt-in smem; >96 KiB reads global env.
    use_static_smem = src.cuda and env_bytes <= 48 * 1024
    use_dynamic_smem = src.cuda and 48 * 1024 < env_bytes <= 96 * 1024
    use_smem = use_static_smem or use_dynamic_smem
    groups = _packed_linear(src, name, layer["linear"], d3, mul)
    paths = _paths(tp)
    _contractions(src, name, paths, d1, d2, d3, ns)
    pointers = (("const Real*", "features"), ("const Real*", "env"))
    loc = f"const Real* f = features + (long long)e * {fw};\n"
    env_pointer = f"const Real* v = env + (long long)centers[e] * {ew};\n"
    setup = loc + env_pointer
    i0 = "0" if mul1 == 1 else f"u * {d1}"
    args = (
        (("int", "n"),)
        + pointers
        + (
            ("const int*", "centers"),
            ("const int*", "offsets"),
            ("const Real*", "latent"),
            ("int", "width"),
            ("Real*", "next"),
        )
    )
    scalar_body = setup + (
        f"for (int q = OFFSET; q < width + {mul * ns}; q += STEP) {{\n"
        f"Real& dst = next[(long long)e * (width + {mul * ns}) + q];\n"
        "if (q < width) { dst = latent[(long long)e * width + q]; continue; }\n"
        + (
            f"const int u = (q-width) / {ns};\n"
            f"dst = {name}_product((q-width) % {ns}, f + {i0}, v + u * {d2});\n"
            if ns
            else ""
        )
        + "}"
    )

    def serial(body):
        return body.replace("OFFSET", "0").replace("STEP", "1")

    def warp(body):
        return body.replace("OFFSET", "int(threadIdx.x) % 32").replace("STEP", "32")

    src.kernel(
        name + "_scalar",
        args,
        serial(scalar_body),
        grid_body=warp(scalar_body),
        warp_rows=True,
    )
    # Emit directly in grouped GEMM order. 2 CTAs/center halve redundant global
    # env loads into shared memory while maintaining full SM wave saturation.
    pack = []
    if layer["linear"] is None:
        pack.append(
            f"for (int q = OFFSET; q < {cw}; q += STEP) {{\n"
            f"const int k = q % {d3}, u = q / {d3};\n"
            f"product[(long long)e * {cw} + q] = {name}_product(k, f + {i0}, v + u * {d2});\n}}"
        )
    else:
        for dim, _, start, _, count, pi, _ in groups:
            pack.append(
                f"for (int q = OFFSET; q < {dim * mul * count}; q += STEP) {{\n"
                f"const int j = q / {mul * count}, u = q % {mul}, c = q / {mul} % {count};\n"
                f"const int k = {start} + c * {dim} + j;\n"
                f"product[(long long)n * {pi} + ((long long)e * {dim} + j) * {mul * count} + u * {count} + c] = "
                f"{name}_product(k, f + {i0}, v + u * {d2});\n}}"
            )
    if use_smem:
        on_center = (
            f"for (int q = int(threadIdx.x); q < {ew}; q += int(blockDim.x)) "
            f"scratch[q] = env[(long long)(e / 2) * {ew} + q];\n"
            "__syncthreads();\nconst int a = e / 2, stripe = e % 2;\n"
            "for (int e = offsets[a] + 4 * stripe + int(threadIdx.x) / 32; e < offsets[a + 1]; e += 8) {\n"
            + loc
            + "const Real* v = scratch;\n"
            + warp(scalar_body[len(setup) :] + "\n" + "\n".join(pack))
            + "\n}"
        )
    else:
        on_center = (
            "const int a = e / 2, stripe = e % 2;\n"
            f"const Real* v = env + (long long)a * {ew};\n"
            "for (int e = offsets[a] + 4 * stripe + int(threadIdx.x) / 32; e < offsets[a + 1]; e += 8) {\n"
            + loc
            + warp(scalar_body[len(setup) :] + "\n" + "\n".join(pack))
            + "\n}"
        )
    src.kernel(
        name + "_packed",
        args + (("Real*", "product"),),
        serial(scalar_body + "\n" + "\n".join(pack)),
        grid_body=on_center,
        shared=ew if use_static_smem else 0,
        dynamic_shared=use_dynamic_smem,
    )

    common = (
        (("int", "n"),)
        + pointers
        + (
            ("const int*", "centers"),
            ("const Real*", "latent_seed"),
            ("int", "width"),
            ("Real*", "dlatent"),
        )
    )
    seed = f"const Real* drow = latent_seed + (long long)e * (width + {mul * ns});\n"
    dlat = "for (int i = OFFSET; i < width; i += STEP) dlatent[(long long)e * width + i] += drow[i];\n"

    def _const_seed_expr(k):
        if layer["linear"] is None:
            p_expr = f"product[(long long)e * {d3 * mul} + u * {d3} + {k}]"
        else:
            terms = []
            for dim, _, start, stop, count, pi, _ in groups:
                if start <= k < stop:
                    c = (k - start) // dim
                    j = (k - start) % dim
                    terms.append(
                        f"product[(long long)n * {pi} + ((long long)e * {dim} + {j}) "
                        f"* {mul * count} + u * {count} + {c}]"
                    )
            if not terms:
                p_expr = "Real(0)"
            elif len(terms) == 1:
                p_expr = terms[0]
            else:
                p_expr = "(" + " + ".join(terms) + ")"
        if k < ns:
            return f"({p_expr} + drow[width + u * {ns} + {k}])"
        return p_expr

    seed_exprs = [_const_seed_expr(k) for k in range(d3)]
    for left in (True, False):
        direction, size = ("left", d1) if left else ("right", d2)
        components = [[] for _ in range(size)]
        for i, j, k, w in paths:
            components[i if left else j].append(
                f"sum += ({seed_exprs[k]} * {src.real(w)}) * "
                f"operand[{j if left else i}];"
            )
        cases = "\n".join(
            f"case {k}: " + " ".join(ops) + " break;"
            for k, ops in enumerate(components)
            if ops
        )
        src.parts.append(
            f"SP2_INLINE Real {name}_{direction}_packed(int k, int u, int e, int n, "
            "const Real* product, const Real* drow, int width, const Real* operand, Real sum) {\n"
            "switch(k) {\n" + cases + "\n} return sum;\n}"
        )
    for scalar, suf in ((True, "_scalar"), (False, "")):
        getter = (
            f"{name}_left_scalar(i, drow + width + u * {ns}, v + u * {d2}, sum)"
            if scalar
            else f"{name}_left_packed(i, u, e, n, product, drow, width, v + u * {d2}, sum)"
        )
        rev = (
            env_pointer
            + seed
            + dlat
            + (
                f"for (int q = OFFSET; q < {fw}; q += STEP) {{\n"
                f"const int i = q / {mul1}; Real sum = 0;\n"
                f"for (int u = {'0' if mul1 == 1 else f'q % {mul1}'}; "
                f"u < {mul if mul1 == 1 else f'q % {mul1} + 1'}; ++u)\n"
                f"sum = {getter};\n"
                f"df[(long long)e * {fw} + (q % {mul1}) * {d1} + i] = sum;\n}}"
            )
        )
        gpu_reverse = warp(rev)
        if src.cuda and not scalar and mul1 == mul and src.precision == "float32":
            if mul == 16 and len(paths) < 200:
                cost = [0] * d1
                for i, _, _, _ in paths:
                    cost[i] += 1
                packs = [[], []]
                loads = [0, 0]
                for i in sorted(range(d1), key=cost.__getitem__, reverse=True):
                    side = int(loads[1] < loads[0])
                    packs[side].append(i)
                    loads[side] += max(1, cost[i])

                def fixed_left(indices):
                    return "\n".join(
                        f"df[(long long)e * {fw} + u * {d1} + {i}] = "
                        f"{name}_left_packed({i},u,e,n,product,drow,width,v + u * {d2},Real(0));"
                        for i in indices
                    )

                gpu_reverse = (
                    env_pointer
                    + seed
                    + warp(dlat)
                    + "const int u = threadIdx.x % 16;\n"
                    + "if (threadIdx.x % 32 < 16) {\n"
                    + fixed_left(packs[0])
                    + "\n} else {\n"
                    + fixed_left(packs[1])
                    + "\n}"
                )
            elif 16 <= mul <= 32:
                # One warp lane per multiplicity (supports mul=16 heavy l_max>=3, mul=24 soc-e2ip, mul=32 soc-prod).
                # Loop over i for large path counts (>=200) to avoid register spill and long NVRTC compile.
                if len(paths) < 200:
                    fixed_all = "\n".join(
                        f"df[(long long)e * {fw} + u * {d1} + {i}] = "
                        f"{name}_left_packed({i},u,e,n,product,drow,width,v + u * {d2},Real(0));"
                        for i in range(d1)
                    )
                else:
                    fixed_all = (
                        f"for (int i = 0; i < {d1}; ++i)\n"
                        f"  df[(long long)e * {fw} + u * {d1} + i] = "
                        f"{name}_left_packed(i,u,e,n,product,drow,width,v + u * {d2},Real(0));"
                    )
                gpu_reverse = (
                    env_pointer
                    + seed
                    + warp(dlat)
                    + "const int u = int(threadIdx.x) & 31;\n"
                    + (f"if (u < {mul}) {{\n" if mul < 32 else "{\n")
                    + fixed_all
                    + "\n}"
                )
        src.kernel(
            f"{name}_adjoint{suf}",
            common + (("const Real*", "product"), ("Real*", "df")),
            serial(rev),
            grid_body=gpu_reverse,
            warp_rows=True,
            heavy=(not scalar and len(paths) >= 200),
        )
        drow = f"const Real* drow = seed + (long long)e * (width + {mul * ns});\n"
        get_right = (
            f"{name}_right_scalar(j, drow + width + u * {ns}, "
            f"features + (long long)e * {fw} + {i0}, sum)"
            if scalar
            else f"{name}_right_packed(j, u, e, rows, product, drow, width, "
            f"features + (long long)e * {fw} + {i0}, sum)"
        )
        terms = "{\n" + drow + f"sum = {get_right};\n}}\n"
        body = (
            f"const int a = idx / {ew}, j = idx % {d2}, u = idx % {ew} / {d2};\n"
            "Real sum = 0;\n"
            "for(int e = offsets[a]; e < offsets[a+1]; ++e)\n"
            + terms
            + "dv[idx] = sum;"
        )
        groups_env = (ew + 31) // 32
        parallel_body = (
            f"const int a = int(blockIdx.x) / {groups_env};\n"
            "const int lane = int(threadIdx.x) % 32, warp = int(threadIdx.x) / 32;\n"
            f"const int at = int(blockIdx.x) % {groups_env} * 32 + lane;\n"
            f"const int j = at / {mul}, u = at % {mul};\n"
            "Real sum = 0;\n"
            f"if (at < {ew})\n"
            "for (int e = offsets[a] + warp; e < offsets[a + 1]; e += 4)\n"
            + terms
            + "scratch[threadIdx.x] = sum;\n__syncthreads();\n"
            f"if (!warp && at < {ew}) dv[(long long)a * {ew} + u * {d2} + j] = "
            "(sum + scratch[lane + 32]) + (scratch[lane + 64] + scratch[lane + 96]);"
        )
        parallel = src.cuda and src.precision == "float32"
        src.kernel(
            f"{name}_environment{suf}",
            (
                ("int", "n"),
                ("int", "rows"),
                ("const Real*", "seed"),
                ("const Real*", "product"),
                ("const Real*", "features"),
                ("const int*", "offsets"),
                ("int", "width"),
                ("Real*", "dv"),
            ),
            body,
            grid_body=parallel_body if parallel else None,
            shared=128 if parallel else 0,
            heavy=(not scalar and len(paths) >= 200),
        )


def _weighted_env_kernels(src, name, spec):
    """Generalized JIT kernels for weight_initial, environment, and reverse_environment
    supporting arbitrary (mul, D, G) including mul=3, 16, 24, 32, 64."""
    ch = spec["weighted_channels"]
    mul = int(ch["multiplicity"])
    irreps = int(ch["irreps"])
    wi = [int(x) for x in ch["w_index"]]
    dim = len(wi)
    norm_arr = spec["env_sum_normalizations"]
    n_species = int(norm_arr.shape[1]) if norm_arr.ndim == 2 else 0

    by_irrep = [[] for _ in range(irreps)]
    for p, g in enumerate(wi):
        by_irrep[g].append(p)
    members = []
    starts = [0]
    for grp in by_irrep:
        members.extend(grp)
        starts.append(len(members))

    qual = "__device__ const int" if src.cuda else "static const int"
    src.parts.append(
        f"{qual} {name}_windex[{dim}] = {{" + ", ".join(str(x) for x in wi) + "};\n"
        f"{qual} {name}_members[{len(members)}] = {{"
        + ", ".join(str(x) for x in members)
        + "};\n"
        f"{qual} {name}_starts[{irreps + 1}] = {{"
        + ", ".join(str(x) for x in starts)
        + "};"
    )

    # 1. weight_initial: out[e, u*D + p] = cutoff[e] > 0 ? attrs[e, p] * weights[e, u*G + wi[p]] : 0
    # Note: weight_initial is only called when embed_initial_edge && layer == 0, where weights has 2*mul*G columns.
    wi_body = (
        f"const bool on = cutoff[e] > Real(0);\n"
        f"const Real* arow = attrs + (long long)e * {dim};\n"
        f"const Real* wrow = weights + (long long)e * {2 * mul * irreps};\n"
        f"Real* orow = out + (long long)e * {mul * dim};\n"
        f"for (int q = OFFSET; q < {mul * dim}; q += STEP) {{\n"
        f"const int u = q / {dim}, p = q - u * {dim};\n"
        f"orow[q] = on ? arow[p] * wrow[u * {irreps} + {name}_windex[p]] : Real(0);\n}}"
    )
    src.kernel(
        name + "_weight_initial",
        (
            ("int", "n"),
            ("const Real*", "attrs"),
            ("const Real*", "weights"),
            ("const Real*", "cutoff"),
            ("Real*", "out"),
        ),
        wi_body.replace("OFFSET", "0").replace("STEP", "1"),
        grid_body=wi_body.replace("OFFSET", "int(threadIdx.x) & 31").replace(
            "STEP", "32"
        ),
        warp_rows=True,
    )

    # 2. environment forward: out[a, u*D + p] = norm * sum_{e in a, cut>0} attrs[e,p] * weights[e, skip + u*G + wi[p]]
    ew = mul * dim
    norm_expr = (
        f"norm[layer * {n_species} + species[a]]" if n_species else "norm[layer]"
    )
    env_serial = (
        f"const int a = idx / {ew}, rem = idx - a * {ew}, u = rem / {dim}, p = rem - u * {dim};\n"
        f"const int col = skip + u * {irreps} + {name}_windex[p];\n"
        f"const int wcols = skip + {mul * irreps};\n"
        "Real sum = 0;\n"
        "for (int e = offsets[a]; e < offsets[a + 1]; ++e)\n"
        f"  if (cutoff[e] > Real(0)) sum += attrs[(long long)e * {dim} + p] * weights[(long long)e * wcols + col];\n"
        f"out[idx] = sum * {norm_expr};"
    )
    groups_env = (ew + 31) // 32
    env_parallel = (
        f"const int a = int(blockIdx.x) / {groups_env};\n"
        "const int lane = int(threadIdx.x) & 31, warp = int(threadIdx.x) >> 5;\n"
        f"const int at = (int(blockIdx.x) % {groups_env}) * 32 + lane;\n"
        f"const int u = at / {dim}, p = at - u * {dim};\n"
        f"const int wcols = skip + {mul * irreps};\n"
        "Real sum = 0;\n"
        f"if (at < {ew}) {{\n"
        f"  const int col = skip + u * {irreps} + {name}_windex[p];\n"
        "  for (int e = offsets[a] + warp; e < offsets[a + 1]; e += 4)\n"
        f"    if (cutoff[e] > Real(0)) sum += attrs[(long long)e * {dim} + p] * weights[(long long)e * wcols + col];\n"
        "}\n"
        "scratch[threadIdx.x] = sum;\n__syncthreads();\n"
        f"if (!warp && at < {ew})\n"
        f"  out[(long long)a * {ew} + at] = ((sum + scratch[lane + 32]) + (scratch[lane + 64] + scratch[lane + 96])) * {norm_expr};"
    )
    env_par = src.cuda and src.precision == "float32"
    src.kernel(
        name + "_env_fwd",
        (
            ("int", "n"),
            ("const Real*", "attrs"),
            ("const Real*", "weights"),
            ("const Real*", "cutoff"),
            ("const int*", "offsets"),
            ("const int*", "species"),
            ("const Real*", "norm"),
            ("int", "layer"),
            ("int", "skip"),
            ("Real*", "out"),
        ),
        env_serial,
        grid_body=env_parallel if env_par else None,
        shared=128 if env_par else 0,
    )

    # 3. reverse_environment: fused d_attrs + d_weights + cutoff mask on d_features (for ANY mul, D, G!)
    rev_env_body = (
        f"const int skip = is_initial ? {mul * irreps} : 0;\n"
        f"const int wcols = skip + {mul * irreps};\n"
        "if (cutoff[e] <= Real(0)) {\n"
        "  for (int c = OFFSET; c < wcols; c += STEP) d_weights[(long long)e * wcols + c] = Real(0);\n"
        "  if (layer != 0)\n"
        "    for (int c = OFFSET; c < dfeat_cols; c += STEP) d_features[(long long)e * dfeat_cols + c] = Real(0);\n"
        "  EARLY_EXIT;\n}\n"
        "const int a = center[e];\n"
        f"const Real scale = {norm_expr};\n"
        f"const Real* de_row = d_env + (long long)a * {ew};\n"
        f"const Real* df_row = d_features + (long long)e * dfeat_cols;\n"
        f"const Real* w_row = weights + (long long)e * wcols;\n"
        f"const Real* a_row = attrs + (long long)e * {dim};\n"
        f"for (int p = OFFSET; p < {dim}; p += STEP) {{\n"
        f"  const int group = {name}_windex[p];\n"
        "  Real sum = 0;\n"
        f"  for (int u = 0; u < {mul}; ++u) {{\n"
        f"    sum += (de_row[u * {dim} + p] * scale) * w_row[skip + u * {irreps} + group];\n"
        f"    if (is_initial) sum += df_row[u * {dim} + p] * w_row[u * {irreps} + group];\n"
        "  }\n"
        "  if (layer == 0 && !is_initial) sum += df_row[p];\n"
        f"  d_attrs[(long long)e * {dim} + p] += sum;\n"
        "}\n"
        f"for (int c = OFFSET; c < {mul * irreps}; c += STEP) {{\n"
        f"  const int u = c / {irreps}, group = c - u * {irreps}, off = u * {dim};\n"
        "  Real initial_sum = 0, env_sum = 0;\n"
        f"  for (int k = {name}_starts[group]; k < {name}_starts[group + 1]; ++k) {{\n"
        f"    const int p = {name}_members[k];\n"
        "    const Real av = a_row[p];\n"
        "    env_sum += (de_row[off + p] * scale) * av;\n"
        "    if (is_initial) initial_sum += df_row[off + p] * av;\n"
        "  }\n"
        "  if (is_initial) d_weights[(long long)e * wcols + c] = initial_sum;\n"
        "  d_weights[(long long)e * wcols + skip + c] = env_sum;\n"
        "}"
    )
    src.kernel(
        name + "_rev_env",
        (
            ("int", "n"),
            ("const Real*", "d_env"),
            ("Real*", "d_features"),
            ("const Real*", "cutoff"),
            ("const Real*", "weights"),
            ("const Real*", "attrs"),
            ("const int*", "center"),
            ("const int*", "species"),
            ("const Real*", "norm"),
            ("int", "layer"),
            ("int", "is_initial"),
            ("int", "dfeat_cols"),
            ("Real*", "d_attrs"),
            ("Real*", "d_weights"),
        ),
        rev_env_body.replace("OFFSET", "0")
        .replace("STEP", "1")
        .replace("EARLY_EXIT", "continue"),
        grid_body=rev_env_body.replace("OFFSET", "int(threadIdx.x) & 31")
        .replace("STEP", "32")
        .replace("EARLY_EXIT", "return"),
        warp_rows=True,
    )


def _epilogues(src, name, spec):
    polynomial = int(spec["polynomial_cutoff_p"])
    if spec["cutoff_type"] == "polynomial":
        cutoff = (
            "Real f = 0, df = 0;\n"
            "if (r < radius) {\n"
            "const Real u = r / radius, tail = Real(1) - u;\n"
            f"Real series = Real({polynomial}) * Real({polynomial + 1}) / Real(2);\n"
            f"for (int k = {polynomial - 2}; k >= 0; --k) "
            "series = series * u + Real(k+1)*Real(k+2)/Real(2);\n"
            "f = tail * tail * tail * series;\n"
            f"df = -(Real({polynomial}) * Real({polynomial + 1}) * Real({polynomial + 2}) / Real(2)) * "
            f"sp2_ipow_pos(u, {polynomial - 1}) * tail * tail / radius;\n"
            "}"
        )
    elif spec["cutoff_type"] == "cosine":
        prune = (
            " && f > Real(0)" if spec.get("edge_prune", "nonzero") == "nonzero" else ""
        )
        cutoff = (
            f"const Real start = radius * {src.real(spec['r_start_cos_ratio'])};\n"
            "const Real freq = Real(3.1415926535897932384626433832795) / (radius - start);\n"
            "const Real phase = freq * ((r < start ? start : (r > radius ? radius : r)) - start);\n"
            "const Real f = Real(.5) * (cos(phase) + Real(1));\n"
            f"const Real df = r >= start && r <= radius{prune} ? -Real(.5)*freq*sin(phase) : Real(0);"
        )
    else:
        raise ValueError(f"Unknown backbone cutoff {spec['cutoff_type']!r}")
    body = (
        "const Real r = lengths[e];\n"
        + cutoff
        + "\nif (OFFSET == 0) { cut[e] = f; deriv[e] = df; }\n"
        "for(int c = OFFSET; c < width; c += STEP) {\n"
        "const long long at = (long long)e * width + c;\n"
        "const Real prior = previous ? previous[at] : Real(0);\n"
        "const Real v = f > Real(0) ? update[at] * f : Real(0);\n"
        "out[at] = residual ? old * prior + next * v : (f > Real(0) ? v : prior);\n"
        "}"
    )

    src.kernel(
        name + "_epilogue",
        (
            ("int", "n"),
            ("const Real*", "lengths"),
            ("const Real*", "update"),
            ("const Real*", "previous"),
            ("Real*", "cut"),
            ("Real*", "deriv"),
            ("Real*", "out"),
            ("int", "width"),
            ("Real", "radius"),
            ("int", "residual"),
            ("Real", "old"),
            ("Real", "next"),
        ),
        body.replace("OFFSET", "0").replace("STEP", "1"),
        grid_body=(
            "if (e >= n) return;\n"
            + body.replace("OFFSET", "int(threadIdx.x) % 32").replace("STEP", "32")
        ),
        warp_rows=True,
    )
    arguments = (
        ("int", "n"),
        ("const Real*", "seed"),
        ("const Real*", "additional"),
        ("const Real*", "y"),
        ("const Real*", "cut"),
        ("const Real*", "deriv"),
        ("Real*", "d_lengths"),
        ("Real*", "d_update"),
        ("Real*", "d_previous"),
        ("int", "width"),
        ("int", "residual"),
        ("Real", "old"),
        ("Real", "next"),
    )
    stable = (
        "const Real f = cut[e];\n"
        "const Real a = residual ? old : (f > Real(0) ? Real(0) : Real(1));\n"
        "const Real b = residual ? next : (f > Real(0) ? Real(1) : Real(0));\n"
    )
    body = stable + (
        "Real dc = 0;\n"
        "for (int c = 0; c < width; ++c) {\n"
        "const long long at = (long long)e * width + c;\n"
        "const Real incoming = additional ? seed[at] + additional[at] : seed[at];\n"
        "const Real g = b * incoming;\n"
        "dc += g * y[at];\n"
        "d_update[at] = f * g; d_previous[at] = a * incoming;\n"
        "}\n"
        "if (deriv[e] != Real(0)) d_lengths[e] += dc * deriv[e];"
    )
    gpu = stable + (
        "const int lane = int(threadIdx.x) % 32;\n"
        "Real dc = 0;\n"
        "for (int c = lane; c < width; c += 32) {\n"
        "const long long at = (long long)e * width + c;\n"
        "const Real incoming = additional ? seed[at] + additional[at] : seed[at];\n"
        "const Real g = b * incoming;\n"
        "dc += g * y[at];\n"
        "d_update[at] = f * g; d_previous[at] = a * incoming;\n"
        "}\n"
        "for (int d = 16; d > 0; d /= 2) dc += __shfl_down_sync(0xffffffffu, dc, d);\n"
        "if (!lane && deriv[e] != Real(0)) d_lengths[e] += dc * deriv[e];"
    )
    src.kernel(
        name + "_epilogue_reverse",
        arguments,
        body,
        grid_body=gpu
        if src.precision == "float32"
        else "if (int(threadIdx.x) % 32 == 0) {\n" + body + "\n}",
        warp_rows=True,
    )


def _mlp_kernels(src):
    # cuBLAS/cuBLASLt owns GEMM contraction order; specialize vectorized epilogues.
    args = (("int", "n"), ("Real*", "output"), ("Real*", "slopes"))
    silu_scalar = (
        "const Real x = output[idx];\n"
        "const bool negative = x < Real(0);\n"
        "const Real a = exp(negative ? x : -x), den = Real(1) + a;\n"
        "const Real y = output[idx] = negative ? (x * a) / den : x / den;\n"
        "if (slopes) {\n"
        "const Real sig = negative ? a/den : Real(1)/den;\n"
        "slopes[idx] = sig + y * (Real(1) - sig);\n"
        "}"
    )
    silu_cuda = (
        """const int idx = int(blockIdx.x * blockDim.x + threadIdx.x);
if ((n & 3) == 0 && (((unsigned long long)output & 15ull) == 0ull) &&
    (!slopes || (((unsigned long long)slopes & 15ull) == 0ull))) {
    if (idx >= (n >> 2)) return;
    const float4 xv = reinterpret_cast<const float4*>(output)[idx];
    auto step = [](float x, float& s) {
        const bool neg = x < 0.0f;
        const float a = expf(neg ? x : -x), den = 1.0f + a;
        const float y = neg ? (x * a) / den : x / den;
        const float sig = neg ? a / den : 1.0f / den;
        s = sig + y * (1.0f - sig);
        return y;
    };
    float4 yv, sv;
    yv.x = step(xv.x, sv.x);
    yv.y = step(xv.y, sv.y);
    yv.z = step(xv.z, sv.z);
    yv.w = step(xv.w, sv.w);
    reinterpret_cast<float4*>(output)[idx] = yv;
    if (slopes) reinterpret_cast<float4*>(slopes)[idx] = sv;
    return;
}
if (idx >= n) return;
"""
        + silu_scalar
        if src.cuda and src.precision == "float32"
        else None
    )
    src.kernel("sp2_trunk_hidden_silu", args, silu_scalar, raw_cuda_body=silu_cuda)

    ssp_scalar = (
        "const Real x = output[idx];\n"
        "const bool linear = x > Real(20);\n"
        "const Real e = linear ? Real(0) : exp(x);\n"
        "output[idx] = (linear ? x : log1p(e)) - Real(0.693147180559945309417232121458);\n"
        "if (slopes) slopes[idx] = linear ? Real(1) : e/(Real(1)+e);"
    )
    ssp_cuda = (
        """const int idx = int(blockIdx.x * blockDim.x + threadIdx.x);
if ((n & 3) == 0 && (((unsigned long long)output & 15ull) == 0ull) &&
    (!slopes || (((unsigned long long)slopes & 15ull) == 0ull))) {
    if (idx >= (n >> 2)) return;
    const float4 xv = reinterpret_cast<const float4*>(output)[idx];
    auto step = [](float x, float& s) {
        const bool lin = x > 20.0f;
        const float e = lin ? 0.0f : expf(x);
        s = lin ? 1.0f : e / (1.0f + e);
        return (lin ? x : log1pf(e)) - 0.6931471805599453f;
    };
    float4 yv, sv;
    yv.x = step(xv.x, sv.x);
    yv.y = step(xv.y, sv.y);
    yv.z = step(xv.z, sv.z);
    yv.w = step(xv.w, sv.w);
    reinterpret_cast<float4*>(output)[idx] = yv;
    if (slopes) reinterpret_cast<float4*>(slopes)[idx] = sv;
    return;
}
if (idx >= n) return;
"""
        + ssp_scalar
        if src.cuda and src.precision == "float32"
        else None
    )
    src.kernel("sp2_trunk_hidden_ssp", args, ssp_scalar, raw_cuda_body=ssp_cuda)

    # Vectorized in-place/out-of-place reverse activation multiply (replaces Kokkos scalar loop).
    rmul_scalar = "out[idx] = seed[idx] * slopes[idx];"
    rmul_cuda = (
        """const int idx = int(blockIdx.x * blockDim.x + threadIdx.x);
if ((n & 3) == 0 && (((unsigned long long)seed & 15ull) == 0ull) &&
    (((unsigned long long)slopes & 15ull) == 0ull) &&
    (((unsigned long long)out & 15ull) == 0ull)) {
    if (idx >= (n >> 2)) return;
    const float4 a = reinterpret_cast<const float4*>(seed)[idx];
    const float4 b = reinterpret_cast<const float4*>(slopes)[idx];
    reinterpret_cast<float4*>(out)[idx] = make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
    return;
}
if (idx >= n) return;
out[idx] = seed[idx] * slopes[idx];"""
        if src.cuda and src.precision == "float32"
        else None
    )
    src.kernel(
        "sp2_trunk_reverse_mul",
        (
            ("int", "n"),
            ("const Real*", "seed"),
            ("const Real*", "slopes"),
            ("Real*", "out"),
        ),
        rmul_scalar,
        raw_cuda_body=rmul_cuda,
        no_restrict=("seed", "out"),
    )

    src.kernel(
        "sp2_trunk_scalar_adjoint",
        (
            ("int", "n"),
            ("int", "width"),
            ("const Real*", "seed"),
            ("const Real*", "weights"),
            ("const Real*", "slopes"),
            ("Real*", "out"),
        ),
        "out[idx] = (seed[idx / width] * weights[idx % width]) * slopes[idx];",
    )


def generate_source(model, backend, precision, gpu_arch=None):
    """Return a translation unit and the exhaustive list of required symbols."""
    if backend not in ("cuda", "cpu") or precision not in ("float32", "float64"):
        raise ValueError("Expected CUDA/CPU and IEEE float32/float64")
    src = Source(backend, precision, gpu_arch=gpu_arch or "")
    _mlp_kernels(src)
    for layer in model["layers"]:
        if "backbone" not in layer:
            continue
        name = _name(layer["name"])
        spec = layer["backbone"]
        _epilogues(src, name, spec)
        _weighted_env_kernels(src, name, spec)
        for i, group in enumerate(spec["layers"]):
            _tensor_layer(src, f"{name}_tp{i}", group)

    from .jit_geometry_codegen import generate_geometry

    geometry, symbols = generate_geometry(model, backend, precision, gpu_arch=gpu_arch)
    return "\n\n".join([*src.parts, geometry]), tuple([*src.symbols, *symbols])
