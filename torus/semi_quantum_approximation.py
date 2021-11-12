# Copyright Bartosz Rzepkowski, Michał Kupczyński, Wrocław 2020
from tenpy.networks.site import SpinSite
from tenpy.models.lattice import Honeycomb
from tenpy.models.spins import SpinModel
from tenpy.networks.mps import MPS
import numpy as np
from math import sqrt
import sys

def HoneyComb_model(lattice, Jx, Jy, Jz, D, hx=0., hy=0., hz=0., muJ=0., E=0.):

    model_params = {
        "lattice": lattice,
        "Jx": Jx,
        "Jy": Jy,
        "Jz": Jz,
        "hx": hx,
        "hy": hy,
        "hz": hz,
        "muJ": muJ,
        "D": D,
        "E": E,
        "verbose": False,
    }
    model = SpinModel(model_params)
    return model


def semi_quantum_approximation(Lx, Ly, J, L, D):
    spinSite = SpinSite(S=1.5, conserve=None)
    lattice = Honeycomb(Lx=Lx, Ly=Ly, sites=spinSite, bc=['periodic', 'periodic'], bc_MPS='finite')
    model = HoneyComb_model(lattice=lattice, Jx=-J, Jy=-J, Jz=-J-L, D=-D)
    MPO = model.calc_H_MPO()
    L = lattice.N_sites

    up_state = np.array([1, 0, 0, 0])
    down_state = np.array([0, 0, 0, 1])
    right_state = np.array([1/(2*sqrt(2)), sqrt(3)/(2*sqrt(2)), sqrt(3)/(2*sqrt(2)), 1/(2*sqrt(2))])
    left_state = np.array([1/(2*sqrt(2)), -sqrt(3)/(2*sqrt(2)), sqrt(3)/(2*sqrt(2)), -1/(2*sqrt(2))])

    FM_Z_product_state = [up_state] * L
    AMF_Z_product_state = []
    up = True
    for i in range(2*Lx):
        for j in range(Ly):
            if up:
                AMF_Z_product_state.append(up_state)
            else:
                AMF_Z_product_state.append(down_state)
        up = not up
    # print("FM_Z_product_state: ", FM_Z_product_state)
    # print("AMF_Z_product_state: ", AMF_Z_product_state)

    FM_XY_product_state = [right_state] * L
    AMF_XY_product_state = []
    right = True
    for i in range(2*Lx):
        for j in range(Ly):
            if right:
                AMF_XY_product_state.append(right_state)
            else:
                AMF_XY_product_state.append(left_state)
        right = not right
    # print("FM_XY_product_state: ", FM_XY_product_state)
    # print("AMF_XY_product_state: ", AMF_XY_product_state)

    # product_state = [Sx_positive_state] * L
    FM_Z_psi = MPS.from_product_state(model.lat.mps_sites(), FM_Z_product_state, bc=model.lat.bc_MPS)
    FM_Z_energy = MPO.expectation_value(FM_Z_psi)

    AMF_Z_psi = MPS.from_product_state(model.lat.mps_sites(), AMF_Z_product_state, bc=model.lat.bc_MPS)
    AFM_Z_energy = MPO.expectation_value(AMF_Z_psi)

    FM_XY_psi = MPS.from_product_state(model.lat.mps_sites(), FM_XY_product_state, bc=model.lat.bc_MPS)
    FM_XY_energy = MPO.expectation_value(FM_XY_psi)

    AMF_XY_psi = MPS.from_product_state(model.lat.mps_sites(), AMF_XY_product_state, bc=model.lat.bc_MPS)
    AFM_XY_energy = MPO.expectation_value(AMF_XY_psi)

    # print("FM_Z_energy: ", FM_Z_energy)
    # print("AFM_Z_energy: ", AFM_Z_energy)
    # print("FM_XY_energy: ", FM_XY_energy)
    # print("AFM_XY_energy: ", AFM_XY_energy)

    groundstate_energy = np.min([FM_Z_energy, AFM_Z_energy, FM_XY_energy, AFM_XY_energy])
    # print("groundstate_energy: ", groundstate_energy)

    if groundstate_energy == FM_Z_energy:
        return groundstate_energy, 1
    elif groundstate_energy == AFM_Z_energy:
        return groundstate_energy, 2
    elif groundstate_energy == FM_XY_energy:
        return groundstate_energy, 3
    elif groundstate_energy == AFM_XY_energy:
        return groundstate_energy, 4


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Incorrect number of arguments. Provide them in the following form:")
        print("- Lx,")
        print("- Ly,")
        print("- J - parameter in the Hamiltonian,")
        print("- L - parameter in the Hamiltonian,")
        print("- D - parameter in the Hamiltonian.")
        sys.exit()

    Lx = int(sys.argv[1])
    Ly = int(sys.argv[2])
    J = round(float(sys.argv[3]), 3)
    L = round(float(sys.argv[4]), 3)
    D = round(float(sys.argv[5]), 3)
    print(semi_quantum_approximation(Lx, Ly, J, L, D))
