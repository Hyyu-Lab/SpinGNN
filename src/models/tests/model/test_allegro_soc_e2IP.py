"""Both SOC evidential targets use the current magnetic energy builder."""

import pytest

from test_allegro_e2IP import E2IP_WARNINGS, check_evidence_model


pytestmark = E2IP_WARNINGS


@pytest.mark.parametrize("key", ["forces", "spin_forces_vert"])
def test_allegro_soc_e2ip_evidence_and_gradients(key):
    check_evidence_model(True, key)
