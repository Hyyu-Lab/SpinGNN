# Install and verify T-e3nn

SpinGNN2 uses the time-reversal-aware [T-e3nn fork](https://github.com/Hongyu-yu/T-e3nn)
for v1 SpinGNN++ (`src/core/`, `src/models/`) training compatibility (note that
`spingnn2.symmetry`, `spingnn2.v2`, and `libspingnn2` have zero `e3nn` runtime
dependency). The CPU installer, CUDA installer, and Docker recipe pin v1.0 at
`75334027fffc38f2268db8ab1eed281a18ffdbf2`. It installs as
`e3nn==0.5.0+spin`. Use the [installation guide](../INSTALL.md) for the
standard environment or follow these steps if you need an explicit dependency
checkout.

## Install the pinned source

From the SpinGNN2 repository root, create a separate dependency checkout and
environment. The example uses the CUDA 12.8 wheel. For a CPU environment,
use the [CPU installer](../tools/ci/README.md#install-and-run-cpu-checks).

```bash
export SPINGNN_CHECKOUT="$PWD"
export TE3NN_CHECKOUT="$PWD/validation/vendor/T-e3nn"
export SPINGNN_ENV="$PWD/validation/env/te3nn-check"
export SPINGNN_PYTHON="$SPINGNN_ENV/bin/python"
TE3NN_PIN=75334027fffc38f2268db8ab1eed281a18ffdbf2

mkdir -p "$(dirname "$TE3NN_CHECKOUT")"
git clone --branch v1.0 https://github.com/Hongyu-yu/T-e3nn.git "$TE3NN_CHECKOUT"
git -C "$TE3NN_CHECKOUT" checkout --detach "$TE3NN_PIN"
test "$(git -C "$TE3NN_CHECKOUT" rev-parse 'v1.0^{commit}')" = "$TE3NN_PIN"

uv venv --python 3.10.20 "$SPINGNN_ENV"
uv pip install --python "$SPINGNN_PYTHON" \
  'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$SPINGNN_PYTHON" \
  -r tools/ci/constraints-ci.txt 'setuptools==78.1.0' wheel
uv pip install --python "$SPINGNN_PYTHON" --no-deps --no-build-isolation \
  "$TE3NN_CHECKOUT"
uv pip install --python "$SPINGNN_PYTHON" --no-deps --no-build-isolation \
  -e ./src/core -e ./src/models -e .
```

Install PyTorch first because T-e3nn imports it at build time. The
`--no-deps` options preserve the installed spin-aware distributions. Reinstall
T-e3nn after changing its source checkout. The command installs a snapshot of
T-e3nn and makes the SpinGNN2 packages editable.

## Verify the installed source

Run the check through the [source launcher](SOURCE_CHECKOUT.md). It verifies
the imported versions and compares the installed T-e3nn Python files with the
pinned checkout.

```bash
"$SPINGNN_PYTHON" tools/ci/source_python.py -c 'import sys; exec(sys.stdin.read())' <<'PY'
from hashlib import sha256
import os
from pathlib import Path

import allegro
import e3nn
import nequip
import torch


def source_map(root):
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*.py")
    }


expected = Path(os.environ["TE3NN_CHECKOUT"]).resolve() / "e3nn"
installed = Path(e3nn.__file__).resolve().parent

assert source_map(installed) == source_map(expected)
assert e3nn.__version__ == "0.5.0+spin"
assert torch.__version__ == "2.11.0+cu128"
assert torch.version.cuda == "12.8"
print("T-e3nn:", installed)
print("NequIP:", nequip.__file__)
print("Allegro:", allegro.__file__)
PY
```

Use a separate environment to compare dependency revisions. Export fresh
`.sp2.json` / `.sp2.bin` contracts (`spingnn2-export`) and pre-compile offline
single-arch or multi-GPU `bundle_abi = 1` JIT artifacts
(`spingnn2-prepare-jit-artifact --arch all-gpu`) for native `libspingnn2` and
modern LAMMPS (`lmp_spingnn2`). See the [SpinGNN2 user guide](SPINGNN2_GUIDE.md).

## Optional Linux process memory policy

[`process_thp.py`](../tools/ci/process_thp.py) can disable transparent huge
pages (THP) for a Linux process and the local children that inherit its
policy. This can help when THP promotion increases resident memory. The
default behavior for commands started directly is the system's current
policy. Pass `--policy inherit` to record that policy without changing it.

```bash
"$SPINGNN_PYTHON" tools/ci/process_thp.py \
  --policy disable --record /path/to/run/thp-policy.json -- \
  "$SPINGNN_PYTHON" "$SPINGNN_CHECKOUT/tools/ci/source_python.py" \
  spingnn-train /path/to/config.yaml
```

Choose a new record path for each run. The launcher writes the requested and
effective flags, the command, PID, and script hash, then replaces itself with
the command. Remote MPI ranks need their own policy check.
