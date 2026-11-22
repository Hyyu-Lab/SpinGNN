import numpy as np


def test_benchmark_fixture_is_local_and_complete(BENCHMARK_ROOT):
    benchmark_path = BENCHMARK_ROOT / "aspirin_ccsd-train.npz"

    with np.load(benchmark_path) as data:
        assert data["spingnn_fixture_version"].item() == 1
        assert data["z"].shape == (21,)
        assert data["R"].shape == (1000, 21, 3)
        assert data["F"].shape == data["R"].shape
        assert data["E"].shape == (1000,)
        assert all(np.isfinite(data[key]).all() for key in ("E", "F", "R"))
        assert np.allclose(data["F"].sum(axis=1), 0.0, atol=1e-12)
        assert 20.0 < data["F"].std() < 40.0
        assert np.abs(data["F"][data["F"] != 0.0]).min() >= 20.0
