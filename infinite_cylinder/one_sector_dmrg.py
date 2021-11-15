# Copyright Bartosz Rzepkowski, Michał Kupczyński, Wrocław 2020

import tenpy.linalg.np_conserved as npc
from tenpy.networks.site import SpinSite
from tenpy.models.lattice import Honeycomb
from tenpy.models.spins import SpinModel
from tenpy.networks.mps import MPS
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections import Counter
import math
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

# N - liczba wszystkich spinów,
# Sz - domyslna podprzestrzeń.
def prep_initial_state(N, Sz):
    psi = ["1.5"]*N # Zainicjalizuj stan początkowy ze wszystkimi spinami w górę
    current_spin = N-1
    current_Sz = N*1.5

    while current_spin >= 0 and current_Sz != Sz:
        if float(psi[current_spin]) > -1.5:
            psi[current_spin] = str(float(psi[current_spin]) - 1.0)
            current_Sz -= 1.0
        else:
            current_spin -= 1

    psi =  np.random.permutation(psi)
    # print(psi)
    return psi

# n - liczba stanów wzbudzonych
def run_dmrg(model,lattice,s,chi_max=1000,svd_min=1.e-10,mixer=True):
    N_sites = model.lat.N_sites
    results = []
    sites = model.lat.mps_sites()
    print("Sector Sz: ", s)


    state = prep_initial_state(N_sites, s)
    psi = MPS.from_product_state(sites, state, model.lat.bc_MPS)
    dmrg_params = {
        "trunc_params": {
            "chi_max": chi_max,
            "svd_min": svd_min
        },
        "mixer": mixer,
        'mixer_params': {},
    }
    eng = TwoSiteDMRGEngine(psi, model, dmrg_params)
    energy, psi = eng.run() # Note, that for iDMRG this correspond to ENERGY PER SITE!
    info = eng.sweep_stats
    print("energy: ", energy)

    #  Sz (average and total)
    Sz_per_site = psi.expectation_value("Sz")
    Sz_per_site_str = list_to_string(Sz_per_site)
    average_Sz = np.sum(Sz_per_site) / N_sites
    Sz_total = np.sum(Sz_per_site)

    # Correlations
    nearest_neighbors = get_nearest_neighbors(lattice)
    correlation_Sp_Sm = psi.correlation_function('Sp', 'Sm')
    correlation_Sm_Sp = psi.correlation_function('Sm', 'Sp')
    correlations = (correlation_Sp_Sm + correlation_Sm_Sp) / 2
    first_indices = []
    second_indices = []
    average_correlation = 0
    for (site_1, site_2) in nearest_neighbors:
        average_correlation += correlations[site_1, site_2]
    average_correlation /= len(nearest_neighbors)

    # Max chi
    max_chi = max(info['max_chi'])

    # Calculate the (half-chain) entanglement entropy for all NONTRIVIAL BONDS.
    # Above probably means, that we will not calculate entanglement entropy
    # for bonds connecting only one site to the rest of the system.
    entanglement_entropy =  psi.entanglement_entropy()
    average_entanglement_entropy = np.sum(entanglement_entropy) / len(entanglement_entropy)

    # Save results
    results.append([energy, average_Sz, average_correlation, max_chi, average_entanglement_entropy, Sz_total, Sz_per_site_str])

    print(psi.correlation_function("Sz","Sz"))

    print("results: ", results)
    return results

# Returns list of MPS indices corresponding to coupled sites in the lattice
def get_nearest_neighbors(lattice):
    neighbors = []
    N_sites = lattice.N_sites
    for u1, u2, dx in lattice.pairs['nearest_neighbors']:
        possible_couplings = lattice.possible_couplings(u1, u2, dx)
        for i in range(len(possible_couplings[0])):
            # MPS with bc_MPS = 'infinite' doesn't recognize the situation in which
            # we are encountering e.g. right edge of the system. It tries to find
            # connections with the part of a lattice, which doesn't exist. Below simple
            # check takes care of this problem.
            if possible_couplings[0][i] < N_sites and possible_couplings[1][i] < N_sites:
                neighbors.append((possible_couplings[0][i], possible_couplings[1][i]))
    return neighbors

def save_to_file(results, file):
    line = ""
    for result in results:
        line += str(result) + "\n"
    file.write(line)

def list_to_string(list):
    string = "("
    for entry in list:
        string += str(entry) + " | "
    string = string[:-3]
    string += ")"
    return string

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Incorrect number of arguments. Provide them in the following form:")
        print("- Lx,")
        print("- Ly,")
        print("- D - parameter in the Hamiltonian,")
        print("- J - parameter in the Hamiltonian,")
        print("- L - parameter in the Hamiltonian,")
        print("- Sz - given Sz sector,")
        print("- finished_params_filename - file containing parameters, for which calculations are already finished.")
        sys.exit()

    Lx = int(sys.argv[1])
    Ly = int(sys.argv[2])
    D = round(float(sys.argv[3]), 3)
    J = round(float(sys.argv[4]), 3)
    L = round(float(sys.argv[5]), 3)
    Sz = int(sys.argv[6])
    finished_params_filename = sys.argv[7]

    spinSite = SpinSite(S=1.5, conserve='Sz')
    lattice = Honeycomb(Lx=Lx, Ly=Ly, sites=spinSite, bc=['periodic', 'periodic'], bc_MPS='infinite')
    filename = "partial_data_Lx=" + str(Lx) + "_Ly=" + str(Ly) + "_D=" + str(D) + "_J=" + str(J) + "_L=" + str(L) + "_Sz=" + str(Sz) + ".csv"
    file = open(filename,"w+")

    #===========================================================================
    print("#########################################")
    print("###### D: ", D, ", J: ", J, ", L: ", L)
    print("#########################################")
    # Note, that below we are passing negative values of the above parameter, because
    # the defnition of Hamiltonian in SpinModel is different from the Hamiltonian defined
    # in the article, that we are basing on.
    model = HoneyComb_model(lattice=lattice, Jx=-J, Jy=-J, Jz=-J-L, D=-D)
    results = run_dmrg(model=model,lattice=lattice,s=Sz,chi_max=2000)
    save_to_file(results, file)

    file = open(finished_params_filename, "a+")
    line = str(Lx) + " " + str(Ly) + " " + str(D) + " " + str(J) + " " + str(L) + " " + str(Sz) + "\n"
    file.write(line)
    file.close()
    #===========================================================================
