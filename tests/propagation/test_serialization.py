"""Unit tests for serialization of optical components.

These tests verify that Optic, Surface, Material, and PropagationModel
instances can be serialized to a dictionary and deserialized back into
fully functional objects, including the correct resolution of all
internal dependencies.
"""

from __future__ import annotations

import pytest
import yaml

from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.standard import StandardGeometry
from optiland.materials.abbe import AbbeMaterial
from optiland.materials.ideal import IdealMaterial
from optiland.materials.material_file import MaterialFile
from optiland.optic.optic import Optic
from optiland.propagation.grin import GRINPropagation
from optiland.propagation.homogeneous import HomogeneousPropagation
from optiland.surfaces.standard_surface import Surface

# --- Test Cases ---
# Each tuple defines a test scenario: (material_instance, expected_propagation_model_class)

# Case 1: IdealMaterial with the default HomogeneousPropagation model.
case_ideal_default = (IdealMaterial(n=1.5, k=0.1), HomogeneousPropagation)

# Case 2: IdealMaterial explicitly assigned a GRINPropagation model.
grin_material = IdealMaterial(n=1.6)
grin_material.propagation_model = GRINPropagation()
case_ideal_grin = (grin_material, GRINPropagation)

# Case 3: AbbeMaterial with the default HomogeneousPropagation model.
case_abbe_default = (
    AbbeMaterial(n=1.5168, abbe=64.17, model="polynomial"),
    HomogeneousPropagation,
)


# --- Helper Function for File-Based Materials ---


def create_dummy_material_file(tmp_path):
    """Creates a temporary material file for testing."""
    content = {
        "DATA": [
            {
                "type": "formula 1",
                "coefficients": "0.6961663 0.4079426 0.04 0.8974794 0.116 1",
            }
        ]
    }
    file_path = tmp_path / "test_material.yml"
    with open(file_path, "w") as f:
        yaml.dump(content, f)
    return str(file_path)


# --- The Main Test Function ---


@pytest.mark.parametrize(
    "material, expected_model_class",
    [
        case_ideal_default,
        case_ideal_grin,
        case_abbe_default,
        # More test cases can be easily added here.
    ],
)
def test_optic_serialization_round_trip(
    material, expected_model_class, set_test_backend
):
    """
    Verify that a complete Optic object can be serialized and deserialized
    correctly, preserving all material and propagation model properties and
    their relationships.
    """
    # 1. SETUP: Create a standard optic with the specified material.
    optic = Optic()
    optic.surfaces.add(
        Surface(
            previous_surface=None,
            geometry=StandardGeometry(coordinate_system=CoordinateSystem(), radius=100),
            material_post=material,
        )
    )

    # 2. ACTION: Perform the serialization and deserialization round-trip.
    optic_dict = optic.to_dict()
    optic_from_dict = Optic.from_dict(optic_dict)

    # 3. VERIFICATION: Check the deserialized object for correctness.
    deserialized_surface = optic_from_dict.surfaces[0]
    deserialized_material = deserialized_surface.material_post
    deserialized_model = deserialized_material.propagation_model

    # Check that the material type is correct.
    assert isinstance(deserialized_material, type(material))

    # Check that the propagation model is the correct type.
    assert isinstance(deserialized_model, expected_model_class)

    # CRITICAL: Check that the propagation model's back-reference to the
    # material was correctly re-established by the factory.
    if hasattr(deserialized_model, "material"):
        assert deserialized_model.material is deserialized_material


def test_material_file_serialization_round_trip(tmp_path, set_test_backend):
    """
    Verify the round-trip serialization for MaterialFile, which requires
    creating a temporary file.
    """
    # 1. SETUP: Create a dummy material file and a MaterialFile instance.
    dummy_file = create_dummy_material_file(tmp_path)
    material = MaterialFile(filename=dummy_file)

    optic = Optic()
    optic.surfaces.add(
        Surface(
            previous_surface=None,
            geometry=StandardGeometry(
                coordinate_system=CoordinateSystem(), radius=-100
            ),
            material_post=material,
        )
    )

    # 2. ACTION: Perform the serialization and deserialization.
    optic_dict = optic.to_dict()
    optic_from_dict = Optic.from_dict(optic_dict)

    # 3. VERIFICATION
    deserialized_material = optic_from_dict.surfaces[0].material_post

    # Check material properties.
    assert isinstance(deserialized_material, MaterialFile)
    assert deserialized_material.filename == dummy_file

    # Check propagation model properties.
    deserialized_model = deserialized_material.propagation_model
    assert isinstance(deserialized_model, HomogeneousPropagation)
    assert deserialized_model.material is deserialized_material
