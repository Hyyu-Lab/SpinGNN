import numpy as np
import pytest

from nequip.utils.data_outcar_spin import read_OSZICAR


def oszicar_text(spin_rows=None, trailing=False):
    spin_rows = spin_rows or ["2 1.0 2.0 3.0", "1 4.0 5.0 6.0"]
    lines = [
        "2 atoms",
        "0 header",
        "0 header",
        "0 header",
        "0 lambda*MW_perp",
        *spin_rows,
    ]
    if trailing:
        lines.extend(["99 91.0 92.0 93.0", "98 81.0 82.0 83.0"])
    lines.extend(
        [
            "0 MW_int",
            "2 0.0 3.0 4.0",
            "1 1.0 0.0 0.0",
        ]
    )
    if trailing:
        lines.append("97 71.0 72.0 73.0")
    return "\n".join(lines) + "\n"


def test_read_oszicar_missing_blocks_has_explicit_policy(tmp_path):
    path = tmp_path / "OSZICAR"
    path.write_text("1 F= -1.0 E0= -1.0\n", encoding="ascii")

    with pytest.raises(ValueError, match="missing required OSZICAR marker"):
        read_OSZICAR(path)
    assert read_OSZICAR(path, on_missing="skip") == (None, None)


def test_read_oszicar_normal_blocks(tmp_path):
    path = tmp_path / "OSZICAR"
    path.write_text(oszicar_text(), encoding="ascii")

    spin_force, mw_int = read_OSZICAR(path, vasp6=True)

    assert np.allclose(spin_force, [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]])
    assert np.allclose(mw_int, [[1.0, 0.0, 0.0], [0.0, 3.0, 4.0]])


def test_read_oszicar_ignores_trailing_rows(tmp_path):
    path = tmp_path / "OSZICAR"
    path.write_text(oszicar_text(trailing=True), encoding="ascii")

    spin_force, mw_int = read_OSZICAR(path)

    assert np.allclose(spin_force, [[8.0, 10.0, 12.0], [2.0, 4.0, 6.0]])
    assert np.allclose(mw_int, [[1.0, 0.0, 0.0], [0.0, 3.0, 4.0]])


@pytest.mark.parametrize(
    ("spin_rows", "error"),
    [
        (["1 1.0 2.0 3.0", "1 4.0 5.0 6.0"], "duplicate atom IDs"),
        (["3 1.0 2.0 3.0", "1 4.0 5.0 6.0"], "atom IDs must be"),
        (["2 nan 2.0 3.0", "1 4.0 5.0 6.0"], "non-finite values"),
    ],
)
def test_read_oszicar_rejects_invalid_atom_rows(tmp_path, spin_rows, error):
    path = tmp_path / "OSZICAR"
    path.write_text(oszicar_text(spin_rows=spin_rows), encoding="ascii")

    with pytest.raises(ValueError, match=error):
        read_OSZICAR(path)
