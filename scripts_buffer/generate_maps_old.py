# Copyright Bartosz Rzepkowski, Michał Kupczyński, Wrocław 2020
# This script collects data from all of the files beginning with "data_" in
# the catalog in which it is located

import tenpy.linalg.np_conserved as npc
from tenpy.networks.site import SpinSite
from tenpy.models.lattice import Honeycomb
from tenpy.models.spins import SpinModel
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import sys
import glob, os
import pprint

def plot_2D_map(data, colormap_name, title, x_min, x_max, x_label, y_min, y_max, y_label, target_filename):
    vmin = np.asarray(data).min()
    vmax = np.asarray(data).max()
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    im = ax.imshow(data, cmap=plt.get_cmap(colormap_name), vmin=vmin, vmax=vmax, extent=(x_min, x_max, y_min, y_max))
    fig.colorbar(im, ax=ax)
    plt.savefig(target_filename)

def plot_2D_map_classical_results(data, colormap_name, title, x_min, x_max, x_label, y_min, y_max, y_label, target_filename):
    vmin = 1
    vmax = 4
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    cax = ax.imshow(data, cmap=plt.get_cmap(colormap_name, 4), vmin=vmin, vmax=vmax, extent=(x_min, x_max, y_min, y_max))

    cbar = fig.colorbar(cax, ticks=[1.375, 2.125, 2.875, 3.625])
    cbar.ax.set_yticklabels(["FM | Z", "AF | Z", "FM | XY", "AF | XY"])

    plt.savefig(target_filename)

# We are creating additional function, in which vmin and vmax are passed as an argument
def plot_2D_map_hard_boundaries(data, colormap_name, title, x_min, x_max, x_label, y_min, y_max, y_label, vmin, vmax, target_filename):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    cax = ax.imshow(data, cmap=plt.get_cmap(colormap_name), vmin=vmin, vmax=vmax, extent=(x_min, x_max, y_min, y_max))
    fig.colorbar(cax, ax=ax)
    plt.savefig(target_filename)

# We are creating additional function, in which vmin and vmax are passed as an argument
def plot_2D_map_hard_boundaries_nonlinear(data, colormap_name, title, x_min, x_max, x_label, y_min, y_max, y_label, vmin, vmax, target_filename, jump):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ticks_labels = list(np.arange(vmin, vmax + jump, jump))
    ticks = list(np.linspace(vmin + (jump/2), vmax - (jump/2), len(ticks_labels)))
    cax = ax.imshow(data, cmap=plt.get_cmap(colormap_name, len(ticks_labels)), vmin=vmin, vmax=vmax, extent=(x_min, x_max, y_min, y_max))
    cbar= fig.colorbar(cax, ticks=ticks)
    cbar.ax.set_yticklabels(ticks_labels)

    plt.savefig(target_filename)

class ParameterValues:
    def __init__(self, min_energy, energy_gap, ground_state_average_Sz,
                    ground_state_Sz_total, ground_state_average_correlation,
                    ground_state_max_chi, ground_state_average_entanglement_entropy,
                    Sz_per_site):
        self.min_energy = min_energy
        self.energy_gap = energy_gap
        self.ground_state_average_Sz = ground_state_average_Sz
        self.ground_state_Sz_total = ground_state_Sz_total
        self.ground_state_average_correlation = ground_state_average_correlation
        self.ground_state_max_chi = ground_state_max_chi
        self.ground_state_average_entanglement_entropy = ground_state_average_entanglement_entropy
        self.Sz_per_site = Sz_per_site

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Incorrect number of arguments. Provide them in the following form:")
        print("- file with all possible parameters (e.g. succ_fin_params.csv).")
        sys.exit()

    # Go through all values of the parameter and save them for future use
    all_D_values = []
    all_J_values = []
    all_L_values = []
    all_Sz_values = []

    filename = sys.argv[1]
    file = open(filename, "r")
    data = file.readlines()
    for entry in data:
        entry = entry.replace("\n", "")
        entry = entry.split(" ")
        D = float(entry[2])
        J = float(entry[3])
        L = float(entry[4])
        Sz = float(entry[5])
        if D not in all_D_values:
            all_D_values.append(D)
        if J not in all_J_values:
            all_J_values.append(J)
        if L not in all_L_values:
            all_L_values.append(L)
        if Sz not in all_Sz_values:
            all_Sz_values.append(Sz)

    all_D_values.sort()
    all_J_values.sort()
    all_L_values.sort()
    all_Sz_values.sort()

    #
    # Create maps, which will allow for appropriate storing of the data
    #
    D_map = {}
    index = 0
    for D in all_D_values:
        D_map[D] = index
        index += 1
    J_map = {}
    index = 0
    for J in all_J_values:
        J_map[J] = index
        index += 1
    L_map = {}
    index = 0
    for L in all_L_values:
        L_map[L] = index
        index += 1
    Sz_map = {}
    index = 0
    for Sz in all_Sz_values:
        Sz_map[Sz] = index
        index += 1

    # print("D_map: ", D_map)
    # print("J_map: ", J_map)
    # print("L_map: ", L_map)
    # print("Sz_map: ", Sz_map)

    all_data = [[[None for l in range(len(all_L_values))] for j in range(len(all_J_values))] for d in range(len(all_D_values))]
    # all_data = [[[ParameterValues() for d in range(len(all_D_values))] for j in range(len(all_J_values))] for l in range(len(all_L_values))]
    # print("all_data:")
    # pprint.pprint(all_data)

    # print("D: ", range(len(all_D_values)))
    # print("J: ", range(len(all_J_values)))
    # print("L: ", range(len(all_L_values)))
    # for d in range(len(all_D_values)):
    #     for j in range(len(all_J_values)):
    #         for l in range(len(all_L_values)):
    #             print("d: ", d, ", j: ", j, ", l: ", l)
    #             print(all_data[d][j][l])

    os.chdir("./")
    for filename in glob.glob("data_*"):
        file = open(filename, "r")
        data = file.readlines()
        first_entry = data[0].replace("\n", "").split("\t")
        D = float(first_entry[0])
        J = float(first_entry[1])
        L = float(first_entry[2])

        results = []
        for i in range(1, len(data)):
            result = data[i].replace("\n", "").replace("[", "").replace("]", "").split(", ")
            [energy, average_Sz, average_correlation, max_chi, average_entanglement_entropy,
                Sz_total, Sz_per_site] = [float(result[0]), float(result[1]), float(result[2]), float(result[3]), float(result[4]), float(result[5]), result[6]]

            results.append([energy, average_Sz, average_correlation, max_chi, average_entanglement_entropy, Sz_total, Sz_per_site])

        # Sort results, so that the ones with lowest energy are at the beginning of the list
        results.sort(key = lambda results: results[0])
        # print("###########################")
        # print("## Ground state's spins: ##")
        # print("D: ", D, ", J: ", J, ", L: ", L)
        # print(results[0][0], " -> ", results[0][6])

        # Calculate energy-gap
        [ground_state_energy, ground_state_average_Sz, ground_state_average_correlation,
            ground_state_max_chi, ground_state_average_entanglement_entropy,
            ground_state_Sz_total, Sz_per_site] = results[0]
        min_energy = ground_state_energy
        energy_gap = 0.0
        for result in results:
            if not np.isclose(result[0], min_energy):
                energy_gap = result[0] - min_energy
                break

        # Save results to generate maps
        all_data[D_map[D]][J_map[J]][L_map[L]] = ParameterValues(min_energy,
            energy_gap, ground_state_average_Sz, ground_state_Sz_total, ground_state_average_correlation,
            ground_state_max_chi, ground_state_average_entanglement_entropy, Sz_per_site)

        file.close()

    groundstate_energy_min = np.inf
    groundstate_energy_max = -np.inf
    energy_gap_min = np.inf
    energy_gap_max = -np.inf
    average_Sz_min = np.inf
    average_Sz_max = -np.inf
    total_Sz_min = np.inf
    total_Sz_max = -np.inf
    average_correlations_min = np.inf
    average_correlations_max = -np.inf
    max_chi_min = np.inf
    max_chi_max = -np.inf
    average_entanglement_min = np.inf
    average_entanglement_max = -np.inf
    classical_groundstate_energy_min = np.inf
    classical_groundstate_energy_max = -np.inf
    classical_quantum_differences_min = np.inf
    classical_quantum_differences_max = -np.inf

    for D in all_D_values:
        d = D_map[D]
        all_min_energies = [[0.0 for l in range(len(all_L_values))] for j in range(len(all_J_values))]
        all_energy_gaps = [[0.0 for l in range(len(all_L_values))] for j in range(len(all_J_values))]
        all_average_Sz = [[0.0 for l in range(len(all_L_values))] for j in range(len(all_J_values))]
        all_total_Sz = [[0.0 for l in range(len(all_L_values))] for j in range(len(all_J_values))]
        all_average_correlations = [[0.0 for l in range(len(all_L_values))] for j in range(len(all_J_values))]
        all_max_chis = [[0.0 for l in range(len(all_L_values))] for j in range(len(all_J_values))]
        all_average_entanglement_entropies = [[0.0 for l in range(len(all_L_values))] for j in range(len(all_J_values))]
        all_classical_ground_state_phases = [[0.0 for l in range(len(all_L_values))] for j in range(len(all_J_values))]
        all_classical_ground_state_energies = [[0.0 for l in range(len(all_L_values))] for j in range(len(all_J_values))]
        all_differences_classical_quantum_energies = [[0.0 for l in range(len(all_L_values))] for j in range(len(all_J_values))]
        for J in all_J_values:
            j = J_map[J]
            for L in all_L_values:
                l = L_map[L]
                # print("D: ", D, ", J: ", J, "J_map[J]: ", j, ", L: ", L, ", L_map[L]: ", l, end="")
                parameterValues = all_data[d][j][l]
                all_min_energies[j][l] = parameterValues.min_energy
                all_energy_gaps[j][l] = parameterValues.energy_gap
                all_average_Sz[j][l] = parameterValues.ground_state_average_Sz
                all_total_Sz[j][l] = parameterValues.ground_state_Sz_total
                all_average_correlations[j][l] = parameterValues.ground_state_average_correlation
                all_max_chis[j][l] = parameterValues.ground_state_max_chi
                all_average_entanglement_entropies[j][l] = parameterValues.ground_state_average_entanglement_entropy

                # Generate results predicted by the classical theory (given via analytical functions).
                # Note, that there is no S value in below functions, because it only scales the results.
                E_FM_z = -2*D - 3*(J+L) # Ferromagnet in the "z" axis
                E_AF_z = -2*D + 3*(J+L) # Antiferromagnet in the "z" axis
                E_FM_xy = -3*J # Ferromagnet in the "xy" plane
                E_AF_xy = 3*J # Antiferromagnet in the "xy" plane
                classical_energies = [E_FM_z, E_AF_z, E_FM_xy, E_AF_xy]
                classical_ground_state_energy = min(classical_energies)
                if classical_ground_state_energy == E_FM_z:
                    all_classical_ground_state_phases[j][l] = 1
                elif classical_ground_state_energy == E_AF_z:
                    all_classical_ground_state_phases[j][l] = 2
                elif classical_ground_state_energy == E_FM_xy:
                    all_classical_ground_state_phases[j][l] = 3
                elif classical_ground_state_energy == E_AF_xy:
                    all_classical_ground_state_phases[j][l] = 4

                all_classical_ground_state_energies[j][l] = classical_ground_state_energy
                all_differences_classical_quantum_energies[j][l] = classical_ground_state_energy - parameterValues.min_energy

                # print(" -> ", all_classical_ground_state_phases[j][l])

        min_L = all_L_values[0]
        max_L = all_L_values[-1]
        min_J = all_J_values[0]
        max_J = all_J_values[-1]

        # WARNING!
        # We need to transform the data structures storing our values, because in this form
        # they will be printed in an unintuitive way. To do that we have to:
        # 1) Reverse each row in 2D arrays
        # 2) Transpose the final results (it is done later, when we are passing data to the "plot_2D_map" function)
        for row in all_min_energies:
            row.reverse()
        for row in all_energy_gaps:
            row.reverse()
        for row in all_average_Sz:
            row.reverse()
        for row in all_total_Sz:
            row.reverse()
        for row in all_average_correlations:
            row.reverse()
        for row in all_max_chis:
            row.reverse()
        for row in all_average_entanglement_entropies:
            row.reverse()
        for row in all_classical_ground_state_phases:
            row.reverse()

        # Generate maps
        plot_2D_map(np.asarray(all_min_energies).T, 'viridis', "Ground state energy", min_J, max_J, "J", min_L, max_L, "L", "D="+str(D)+"_ground_state_energy.png")
        plot_2D_map(np.asarray(all_energy_gaps).T, 'magma', "Energy gap", min_J, max_J, "J", min_L, max_L, "L", "D="+str(D)+"_energy_gap.png")
        plot_2D_map_hard_boundaries(np.asarray(all_average_Sz).T, 'inferno', "Average value of $S_z$", min_J, max_J, "J", min_L, max_L, "L", -1.5, 1.5, "D="+str(D)+"_average_Sz.png")
        # plot_2D_map(np.asarray(all_total_Sz).T, 'magma', "Total value of $S_z$", min_J, max_J, "J", min_L, max_L, "L", "D="+str(D)+"_total_Sz.png")
        plot_2D_map(np.asarray(all_average_correlations).T, 'cividis', "Average correlation between nodes ''in-plane''", min_J, max_J, "J", min_L, max_L, "L", "D="+str(D)+"_average_correlation.png")
        plot_2D_map(np.asarray(all_max_chis).T, 'magma', "Max $\chi$", min_J, max_J, "J", min_L, max_L, "L", "D="+str(D)+"_max_chi.png")
        plot_2D_map(np.asarray(all_average_entanglement_entropies).T, 'viridis', "Average entanglement entropy", min_J, max_J, "J", min_L, max_L, "L", "D="+str(D)+"_average_entanglement_entropy.png")
        plot_2D_map_classical_results(np.asarray(all_classical_ground_state_phases).T, 'viridis', "Classical ground state phsae", min_J, max_J, "J", min_L, max_L, "L", "D="+str(D)+"_classical_groundstate_phase.png")
        plot_2D_map(np.asarray(all_classical_ground_state_energies).T, 'inferno', "Classical groundstate energy", min_J, max_J, "J", min_L, max_L, "L", "D="+str(D)+"_classical_groundstate_energy.png")
        plot_2D_map(np.asarray(all_differences_classical_quantum_energies).T, 'magma', "$E_{classical} - E_{DMRG}$", min_J, max_J, "J", min_L, max_L, "L", "D="+str(D)+"_classical_quantum_differences.png")

        # Generate maps (version with hard-coded vmin and vax, so that the plots are scaled appropriately)
        # plot_2D_map_hard_boundaries(np.asarray(all_min_energies).T, 'viridis', "Ground state energy", min_J, max_J, "J", min_L, max_L, "L", -12.0, 0.0, "D="+str(D)+"_ground_state_energy.png")
        # plot_2D_map_hard_boundaries(np.asarray(all_energy_gaps).T, 'magma', "Energy gap", min_J, max_J, "J", min_L, max_L, "L", 0.0, 1.7, "D="+str(D)+"_energy_gap.png")
        # plot_2D_map_hard_boundaries(np.asarray(all_average_Sz).T, 'inferno', "Average value of $S_z$", min_J, max_J, "J", min_L, max_L, "L", -1.5, 1.5, "D="+str(D)+"_average_Sz.png")
        plot_2D_map_hard_boundaries_nonlinear(np.asarray(all_total_Sz).T, 'magma', "Total value of $S_z$", min_J, max_J, "J", min_L, max_L, "L", -12, 11, "D="+str(D)+"_total_Sz.png", jump=1)
        # plot_2D_map_hard_boundaries(np.asarray(all_average_correlations).T, 'cividis', "Average correlation between nodes ''in-plane''", min_J, max_J, "J", min_L, max_L, "L", -4.0, 4.0, "D="+str(D)+"_average_correlation.png")
        # plot_2D_map(np.asarray(all_max_chis).T, 'magma', "Max $\chi$", min_J, max_J, "J", min_L, max_L, "L", "D="+str(D)+"_max_chi.png")
        # plot_2D_map_hard_boundaries(np.asarray(all_average_entanglement_entropies).T, 'viridis', "Average entanglement entropy", min_J, max_J, "J", min_L, max_L, "L", 0.0, 1.9, "D="+str(D)+"_average_entanglement_entropy.png")
        # plot_2D_map_classical_results(np.asarray(all_classical_ground_state_phases).T, 'viridis', "Classical ground state phase", min_J, max_J, "J", min_L, max_L, "L", "D="+str(D)+"_classical_groundstate_phase.png")
        # plot_2D_map_hard_boundaries(np.asarray(all_classical_ground_state_energies).T, 'inferno', "Classical groundstate energy", min_J, max_J, "J", min_L, max_L, "L", -3.25, 0.0, "D="+str(D)+"_classical_groundstate_energy.png")
        # plot_2D_map_hard_boundaries(np.asarray(all_differences_classical_quantum_energies).T, 'magma', "$E_{classical} - E_{DMRG}$", min_J, max_J, "J", min_L, max_L, "L", 0.0, 18.0, "D="+str(D)+"_classical_quantum_differences.png")

        if np.min(all_min_energies) < groundstate_energy_min:
            groundstate_energy_min = np.min(all_min_energies)
        if np.max(all_min_energies) > groundstate_energy_max:
            groundstate_energy_max = np.max(all_min_energies)
        if np.min(all_energy_gaps) < energy_gap_min:
            energy_gap_min = np.min(all_energy_gaps)
        if np.max(all_energy_gaps) > energy_gap_max:
            energy_gap_max = np.max(all_energy_gaps)
        if np.min(all_average_Sz) < average_Sz_min:
            average_Sz_min = np.min(all_average_Sz)
        if np.max(all_average_Sz) > average_Sz_max:
            average_Sz_max = np.max(all_average_Sz)
        if np.min(all_total_Sz) < total_Sz_min:
            total_Sz_min = np.min(all_total_Sz)
        if np.max(all_total_Sz) > total_Sz_max:
            total_Sz_max = np.max(all_total_Sz)
        if np.min(all_average_correlations) < average_correlations_min:
            average_correlations_min = np.min(all_average_correlations)
        if np.max(all_average_correlations) > average_correlations_max:
            average_correlations_max = np.max(all_average_correlations)
        if np.min(all_max_chis) < max_chi_min:
            max_chi_min = np.min(all_max_chis)
        if np.max(all_max_chis) > max_chi_max:
            max_chi_max = np.max(all_max_chis)
        if np.min(all_average_entanglement_entropies) < average_entanglement_min:
            average_entanglement_min = np.min(all_average_entanglement_entropies)
        if np.max(all_average_entanglement_entropies) > average_entanglement_max:
            average_entanglement_max = np.max(all_average_entanglement_entropies)
        if np.min(all_classical_ground_state_energies) < classical_groundstate_energy_min:
            classical_groundstate_energy_min = np.min(all_classical_ground_state_energies)
        if np.max(all_classical_ground_state_energies) > classical_groundstate_energy_max:
            classical_groundstate_energy_max = np.max(all_classical_ground_state_energies)
        if np.min(all_differences_classical_quantum_energies) < classical_quantum_differences_min:
            classical_quantum_differences_min = np.min(all_differences_classical_quantum_energies)
        if np.max(all_differences_classical_quantum_energies) > classical_quantum_differences_max:
            classical_quantum_differences_max = np.max(all_differences_classical_quantum_energies)

    print()
    print("#"*50)
    print("Ground state energy: [", groundstate_energy_min, " , ", groundstate_energy_max, "]")
    print("Energy gap: [", energy_gap_min, " , ", energy_gap_max, "]")
    print("Average Sz: [", average_Sz_min, " , ", average_Sz_max, "]")
    print("Total Sz: [", total_Sz_min, " , ", total_Sz_max, "]")
    print("Average correlations: [", average_correlations_min, " , ", average_correlations_max, "]")
    print("Max chi: [", max_chi_min, " , ", max_chi_max, "]")
    print("Average entanglement entropy: [", average_entanglement_min, " , ", average_entanglement_max, "]")
    print("Classical groundstate energy: [", classical_groundstate_energy_min, " , ", classical_groundstate_energy_max, "]")
    print("Classical-quantum energy differences: [", classical_quantum_differences_min, " , ", classical_quantum_differences_max, "]")
    print("#"*50)
