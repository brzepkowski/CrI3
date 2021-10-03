# Copyright Bartosz Rzepkowski, Michał Kupczyński, Wrocław 2020

import tenpy.linalg.np_conserved as npc
from tenpy.networks.site import SpinSite
from tenpy.models.lattice import Honeycomb
from tenpy.models.spins import SpinModel
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections import Counter
import math
import sys

def HoneyComb_lattice_model(lattice, Jx, Jy, Jz, D, hx=0., hy=0., hz=0., muJ=0., E=0., bcy="periodic", bcx="periodic"):

    model_params = {
        # "S": 1.5,  # Spin 3/2
        # "lattice": "Honeycomb",
        "lattice": lattice,
        "bc_MPS": "finite", # Combined with bcx='periodic' and bcy='periodic' it gives a finite system on the torus.
        "bc_y": bcy,
        "bc_x": bcx,
        # "Lx": Lx,  # defines cylinder circumference
        # "Ly": Ly,  # defines cylinder circumference
        "Jx": Jx,
        "Jy": Jy,
        "Jz": Jz,
        "hx": hx,
        "hy": hy,
        "hz": hz,
        "muJ": muJ,
        "D": D,
        "E": E,
        "conserve": "best"   # Heisenberg coupling
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
def run_dmrg(model,lattice,s,n=1,chi_max=1000,svd_min=1.e-10,mixer=True,generate_maps=True):
    N_sites = model.lat.N_sites
    results = []
    # for i in range(3*N_sites+1):
    #     s = ((-1.5)*N_sites)+i
    sites = model.lat.mps_sites()
    excited_states = []
    print("Sector Sz: ", s)
    max_permutations = max_number_of_permutations(prep_initial_state(N_sites ,s))
    print("max_permutations: ", max_permutations)
    for j in range(n):
        print("==============================")
        print("====== State:", j)
        print("==============================")
        State = prep_initial_state(N_sites, s)
        psi = MPS.from_product_state(sites, State, "finite")
        mixer_params = {}
        dmrg_params = {"trunc_params": {"chi_max": chi_max, "svd_min": svd_min}, "mixer": mixer,"mixer_params":mixer_params,
            'combine': True,"orthogonal_to": excited_states}
        info = dmrg.run(psi, model, dmrg_params)

        # Energy
        energy = info['E']

        #  Sz (averae and total)
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
        max_chi = max(info['sweep_statistics']['max_chi'])

        # Calculate the (half-chain) entanglement entropy for all NONTRIVIAL BONDS.
        # Above probably means, that we will not calculate entanglement entropy
        # for bonds connecting only one site to the rest of the system.
        entanglement_entropy =  psi.entanglement_entropy()
        average_entanglement_entropy = np.sum(entanglement_entropy) / len(entanglement_entropy)

        # Save results
        results.append([energy, average_Sz, average_correlation, max_chi, average_entanglement_entropy, Sz_total, Sz_per_site_str])
        excited_states.append(psi)
        if j >= max_permutations - 1:
            print("-----------> Weszło i przerwało <------------")
            break

    results.sort(key = lambda results: results[0])

    print("results: ", results)
    return results

# Returns list of MPS indices corresponding to coupled sites in the lattice
def get_nearest_neighbors(lattice):
    neighbors = []
    for u1, u2, dx in lattice.pairs['nearest_neighbors']:
        possible_couplings = lattice.possible_couplings(u1, u2, dx)
        for i in range(len(possible_couplings[0])):
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

def max_number_of_permutations(list):
    numbers_of_occurrences = Counter(list).values()
    max_number_of_permutations = math.factorial(len(list))
    for number_of_occur in numbers_of_occurrences:
        max_number_of_permutations /= math.factorial(number_of_occur)
    return max_number_of_permutations

if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("Incorrect number of arguments. Provide them in the following form:")
        print("- Lx,")
        print("- Ly,")
        print("- D - parameter in the Hamiltonian,")
        print("- J - parameter in the Hamiltonian,")
        print("- L - parameter in the Hamiltonian,")
        print("- Sz - given Sz sector,")
        print("- n - number of states per sector,")
        print("- finished_params_filename - file containing parameters, for which calculations are already finished.")
        sys.exit()

    Lx = int(sys.argv[1])
    Ly = int(sys.argv[2])
    D = round(float(sys.argv[3]), 3)
    J = round(float(sys.argv[4]), 3)
    L = round(float(sys.argv[5]), 3)
    Sz = int(sys.argv[6])
    n = int(sys.argv[7])
    finished_params_filename = sys.argv[8]

    spinSite = SpinSite(S=1.5, conserve='Sz')
    lattice = Honeycomb(Lx=Lx, Ly=Ly, sites=spinSite, bc='periodic')
    filename = "partial_data_Lx=" + str(Lx) + "_Ly=" + str(Ly) + "_D=" + str(D) + "_J=" + str(J) + "_L=" + str(L) + "_Sz=" + str(Sz) + ".csv"
    file = open(filename,"w+")

    #===========================================================================
    print("#########################################")
    print("###### D: ", D, ", J: ", J, ", L: ", L)
    print("#########################################")
    # Note, that below we are passing negative values of the above parameter, because
    # the defnition of Hamiltonian in SpinModel is different from the Hamiltonian defined
    # in the article, that we are basing on.
    # bcy can take values: "cylinder" / "ladder"
    # bcx can take values "periodic" / "open"
    # model = HoneyComb_lattice_model(lattice=lattice, Jx=-J, Jy=-J, Jz=-J-L, D=-D, bcy="ladder", bcx="periodic")
    model = HoneyComb_lattice_model(lattice=lattice, Jx=-J, Jy=-J, Jz=-J-L, D=-D, bcy="periodic", bcx="periodic")
    results = run_dmrg(model=model,lattice=lattice,s=Sz,n=n)
    # save_to_file(results, D, J, L, file)
    save_to_file(results, file)

    file = open(finished_params_filename, "a+")
    line = str(Lx) + " " + str(Ly) + " " + str(D) + " " + str(J) + " " + str(L) + " " + str(Sz) + "\n"
    file.write(line)
    file.close()
    #===========================================================================

    file.close()
