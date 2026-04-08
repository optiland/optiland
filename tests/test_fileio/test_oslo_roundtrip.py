import numpy as np
import pytest
from optiland.fileio import load_oslo_file, save_oslo_file
from optiland.optic import Optic

def test_oslo_roundtrip_fno(tmp_path):
    # Setup optic with imageFNO
    optic = Optic(name="FNO_Test")
    optic.add_surface(radius=10.0, thickness=5.0, material="N-BK7")
    optic.add_surface(radius=-10.0, thickness=20.0)
    optic.set_aperture(aperture_type="imageFNO", value=8.0)
    
    # Save to OSLO
    oslo_file = tmp_path / "fno_test.len"
    save_oslo_file(optic, str(oslo_file))
    
    # Reload from OSLO
    reloaded_optic = load_oslo_file(str(oslo_file))
    
    # Verify aperture
    assert reloaded_optic.aperture.ap_type == "imageFNO"
    assert np.isclose(reloaded_optic.aperture.value, 8.0)

def test_oslo_roundtrip_nao(tmp_path):
    # Setup optic with objectNA
    optic = Optic(name="NAO_Test")
    optic.add_surface(radius=10.0, thickness=5.0, material="N-BK7")
    optic.add_surface(radius=-10.0, thickness=20.0)
    optic.set_aperture(aperture_type="objectNA", value=0.1)
    
    # Save to OSLO
    oslo_file = tmp_path / "nao_test.len"
    save_oslo_file(optic, str(oslo_file))
    
    # Reload from OSLO
    reloaded_optic = load_oslo_file(str(oslo_file))
    
    # Verify aperture
    assert reloaded_optic.aperture.ap_type == "objectNA"
    assert np.isclose(reloaded_optic.aperture.value, 0.1)
