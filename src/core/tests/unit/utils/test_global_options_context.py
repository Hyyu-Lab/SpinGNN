"""default_dtype save/restore contract for global_options_context."""

import torch

from nequip.utils._global_options import global_options_context


def test_global_options_context_restores_default_dtype():
    """The body may set default_dtype; the previous value must be restored.

    Regression: a float32 config inside the context leaked into the session
    and broke float64 trainer tests run afterwards (full-suite float64 pass:
    TestTrain.test_train/test_restart_training, TestRescale.test_scaling,
    AWL toy training).
    """
    previous = torch.get_default_dtype()
    try:
        for initial in (torch.float32, torch.float64):
            torch.set_default_dtype(initial)
            other = torch.float64 if initial is torch.float32 else torch.float32
            name = "float64" if other is torch.float64 else "float32"
            with global_options_context({"default_dtype": name}):
                assert torch.get_default_dtype() is other
            assert torch.get_default_dtype() is initial
    finally:
        torch.set_default_dtype(previous)
