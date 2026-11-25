import numpy as np
import pytest
import torch

from nequip.model import model_from_config


class ClassScopedModelFixtures:
    @pytest.fixture(
        scope="class",
        params=(
            [torch.device("cuda"), torch.device("cpu")]
            if torch.cuda.is_available()
            else [torch.device("cpu")]
        ),
    )
    @classmethod
    def device(cls, request):
        return request.param

    @pytest.fixture(scope="class")
    @classmethod
    def model(cls, config, device):
        config, out_fields = config
        model = cls.make_model(config, device=device)
        return model, out_fields


class ClassScopedSpinModelFixtures(ClassScopedModelFixtures):
    @pytest.fixture(scope="class")
    @classmethod
    def model_config(cls, config, device):
        config, out_fields = config
        torch.manual_seed(0)
        np.random.seed(0)
        config = config.copy()
        config.update(
            {
                "num_types": 3,
                "types_names": ["H", "C", "O"],
            }
        )
        model = model_from_config(config).to(device)
        return model, out_fields, config
