import pytest

from cem3d.config import MeshConfig, SetupConfig


@pytest.fixture(scope="session")
def mesh_config():
    return SetupConfig(mesh=MeshConfig(size_max=0.25, size_min=0.08))


@pytest.fixture(scope="session")
def mesh_data(mesh_config, tmp_path_factory):
    pytest.importorskip("dolfinx")
    from cem3d.geometry.mesh import CylinderWithElectrodes

    return CylinderWithElectrodes(mesh_config, cache_dir=tmp_path_factory.mktemp("cache")).get()
