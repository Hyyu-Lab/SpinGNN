"""Negative capability and metadata-schema tests at pair_coeff time."""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers import CASES, FIXTURE_ROOT, run_lammps


@pytest.mark.lammps
def test_source_packages_are_loaded_from_current_worktree(source_import_paths):
    repo_root = Path(__file__).resolve().parents[1]
    for name, package in (("nequip", "src/core"), ("allegro", "src/models")):
        expected = repo_root / package / name / "__init__.py"
        assert source_import_paths[name].resolve() == expected.resolve()


@pytest.mark.lammps
@pytest.mark.parametrize(
    ("variant", "input_name", "expected_error"),
    (
        (
            "float64",
            "schema_spin2.in",
            "currently supports only float32 deployed models; model default_dtype is 'float64'",
        ),
        (
            "missing_spin",
            "schema_spin3.in",
            "requires deployed model output key: spin_forces",
        ),
        (
            "invalid_edge_prune",
            "schema_spin2.in",
            "received invalid deployed model metadata edge_prune='invalid'",
        ),
        (
            "missing_metadata",
            "schema_spin2.in",
            "does not appear to be a deployed NequIP model",
        ),
    ),
)
def test_invalid_deploy_schema_is_rejected_at_pair_coeff(
    variant,
    input_name,
    expected_error,
    schema_model_paths,
    lammps_bin,
    lammps_env,
    tmp_path,
):
    result = run_lammps(
        binary=lammps_bin,
        input_path=FIXTURE_ROOT / "in" / input_name,
        workdir=tmp_path,
        env=lammps_env,
        variables={
            "data": CASES["nosoc"].data_path,
            "model": schema_model_paths[variant],
        },
        check=False,
    )
    assert result.returncode != 0
    assert expected_error in result.output
