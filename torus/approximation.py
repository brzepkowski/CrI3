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
    }
    model = SpinModel(model_params)
    return model

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Incorrect number of arguments. Provide them in the following form:")
        print("- Lx,")
        print("- Ly,")
        print("- J - parameter in the Hamiltonian,")
        print("- L - parameter in the Hamiltonian,")
        print("- D - parameter in the Hamiltonian,")
        print("- Sz - given Sz sector. Provide either '0' or 'MAX'.")
        sys.exit()

    Lx = int(sys.argv[1])
    Ly = int(sys.argv[2])
    J = round(float(sys.argv[3]), 3)
    L = round(float(sys.argv[4]), 3)
    D = round(float(sys.argv[5]), 3)
    Sz = sys.argv[6]

    # spinSite = SpinSite(S=1.5, conserve='Sz')
    spinSite = SpinSite(S=1.5, conserve=None)
    lattice = Honeycomb(Lx=Lx, Ly=Ly, sites=spinSite, bc=['periodic', 'periodic'], bc_MPS='finite')
    model = HoneyComb_model(lattice=lattice, Jx=-J, Jy=-J, Jz=-J-L, D=-D)
    L = lattice.N_sites
    Sx_positive_state = np.array([1/(2*sqrt(2)), sqrt(3)/(2*sqrt(2)), sqrt(3)/(2*sqrt(2)), 1/(2*sqrt(2))])
    product_state = [Sx_positive_state] * L
    psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc=model.lat.bc_MPS)

    # print("psi:")
    # print(psi._B)
    # for tensor in psi._B:
    #     print(tensor)

    MPO = model.calc_H_MPO()
    print("expectation value: ", MPO.expectation_value(psi))
