import tempfile
import copy
from ase.calculators.singlepoint import SinglePointCalculator as sp
# from dask_kubernetes import KubeCluster
# from dask.distributed import Client


def calculate(atoms):
    with tempfile.TemporaryDirectory() as tmp_dir:
        atoms.get_calculator().set(directory=tmp_dir)
        sample_energy = atoms.get_potential_energy(apply_constraint=False)
        sample_forces = atoms.get_forces(apply_constraint=False)
        atoms.set_calculator(
            sp(atoms=atoms, energy=sample_energy, forces=sample_forces)
        )
    return atoms


def compute_with_calc(client, images, calc=None):
    """
    images: list of atoms
    """
    if calc is not None:
        for image in images:
            image.set_calculator(copy.deepcopy(calc))
    L = client.map(calculate, images)
    res_images = client.gather(L)
    return res_images