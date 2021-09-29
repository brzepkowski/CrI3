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
from math import isclose
import sys
import glob, os
import pprint

def plot_2D_map(data, colormap_name, title, x_min, x_max, x_label, y_min, y_max, y_label, target_filename, visible_sidebar=False):
    vmin = np.asarray(data).min()
    vmax = np.asarray(data).max()
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    im = ax.imshow(data, cmap=plt.get_cmap(colormap_name), vmin=vmin, vmax=vmax, extent=(x_min, x_max, y_min, y_max))
    if visible_sidebar:
        fig.colorbar(im, ax=ax)

    plt.savefig(target_filename)
    plt.close()

def plot_2D_map_classical_results(data, colormap_name, title, x_min, x_max, x_label, y_min, y_max, y_label, target_filename, visible_sidebar=False):
    vmin = 1
    vmax = 4
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    cax = ax.imshow(data, cmap=plt.get_cmap(colormap_name, 4), vmin=vmin, vmax=vmax, extent=(x_min, x_max, y_min, y_max))
    # Set custom x and y ticks on the axes
    x_label_list = ['-0.5', '-0.25', '0.0', '0.25', '0.5']
    y_label_list = ['-0.5', '-0.25', '0.0', '0.25', '0.5']
    ax.set_xticks([-0.475, -0.2375, 0.0, 0.2375, 0.475])
    ax.set_xticklabels(x_label_list)
    ax.set_yticks([-0.475, -0.2375, 0.0, 0.2375, 0.475])
    ax.set_yticklabels(y_label_list)

    if visible_sidebar:
        cbar = fig.colorbar(cax, ticks=[1.375, 2.125, 2.875, 3.625])
        cbar.ax.set_yticklabels(["FM | Z", "AF | Z", "FM | XY", "AF | XY"])

    plt.savefig(target_filename)
    plt.close()

# We are creating additional function, in which vmin and vmax are passed as an argument
def plot_2D_map_hard_boundaries(data, colormap_name, title, x_min, x_max, x_label, y_min, y_max, y_label, vmin, vmax, target_filename, visible_sidebar=False):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if colormap_name != None:
        cax = ax.imshow(data, cmap=plt.get_cmap(colormap_name), vmin=vmin, vmax=vmax, extent=(x_min, x_max, y_min, y_max))
    else:
        cax = ax.imshow(data, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax, extent=(x_min, x_max, y_min, y_max))

    # Set custom x and y ticks on the axes
    x_label_list = ['-0.5', '-0.25', '0.0', '0.25', '0.5']
    y_label_list = ['-0.5', '-0.25', '0.0', '0.25', '0.5']
    ax.set_xticks([-0.475, -0.2375, 0.0, 0.2375, 0.475])
    ax.set_xticklabels(x_label_list)
    ax.set_yticks([-0.475, -0.2375, 0.0, 0.2375, 0.475])
    ax.set_yticklabels(y_label_list)

    if visible_sidebar:
        fig.colorbar(cax, ax=ax)

    plt.savefig(target_filename)
    plt.close()

# We are creating additional function, in which vmin and vmax are passed as an argument
def plot_2D_map_hard_boundaries_nonlinear(data, colormap_name, title, x_min, x_max, x_label, y_min, y_max, y_label, vmin, vmax, target_filename, jump, visible_sidebar=False):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ticks_labels = list(np.arange(vmin, vmax + jump, jump))
    ticks = list(np.linspace(vmin + (jump/2), vmax - (jump/2), len(ticks_labels)))
    cax = ax.imshow(data, cmap=plt.get_cmap(colormap_name, len(ticks_labels)), vmin=vmin, vmax=vmax, extent=(x_min, x_max, y_min, y_max))

    # Set custom x and y ticks on the axes
    x_label_list = ['-0.5', '-0.25', '0.0', '0.25', '0.5']
    y_label_list = ['-0.5', '-0.25', '0.0', '0.25', '0.5']
    ax.set_xticks([-0.475, -0.2375, 0.0, 0.2375, 0.475])
    ax.set_xticklabels(x_label_list)
    ax.set_yticks([-0.475, -0.2375, 0.0, 0.2375, 0.475])
    ax.set_yticklabels(y_label_list)

    # Reinitialize "ticks" and "ticks_labels", so that the plot are more readable
    ticks_labels = list(np.arange(vmin, vmax + jump, jump*2))
    ticks = list(np.linspace(vmin + (jump/2), vmax - (jump/2), len(ticks_labels)))

    if visible_sidebar:
        cbar= fig.colorbar(cax, ticks=ticks)
        cbar.ax.set_yticklabels(ticks_labels)

    plt.savefig(target_filename)
    plt.close()

def plot_value_vs_lambda_by_J(Xs, Ys, Ds, J, y_label, target_filename, fig, index):
    markers = ['o-', '*-', 'D--', '^-', 's--', 'P-']
    colors = ['purple', 'orange', 'blue', 'red', 'green', 'brown']
    # fig, ax = plt.subplots()
    ax = fig.add_subplot(2, 2, index)

    titles = [None, '(a)', '(b)', '(c)', '(d)']
    ax.set_title(titles[index], loc='left')

    for i in range(len(Ds)):
        if i == 2 or i == 4:
            ax.plot(Xs[i], Ys[i], markers[i], label=r'$D/J = ' + str(Ds[i] / J) + '$', color=colors[i], linewidth=2, alpha=0.5)
        else:
            ax.plot(Xs[i], Ys[i], markers[i], label=r'$D/J = ' + str(Ds[i] / J) + '$', color=colors[i], linewidth=2)
    ax.set_xlabel(r'$ \lambda / J$')
    ax.set_ylabel(y_label)

    if index == 2: # Add legend only on one subplot
        ax.legend()

    # plt.savefig(target_filename)
    # plt.close()
    # return ax

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

    ############################################################################
    X_cross_cut = []

    Y_cross_cut_avg_Sz = []
    Y_cross_cut_avg_correlation = []
    Y_cross_cut_min_energy = []
    Y_cross_cut_energy_gap = []
    Y_cross_cut_Sz_total = []
    Y_cross_cut_max_chi = []
    Y_cross_cut_avg_entanglement_entropy = []
    Y_cross_cut_Sz_per_site = []

    Ds_cross_cut = []
    ############################################################################

    D_max = max(all_D_values)
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

            ########################################################################
            X = []
            Y_avg_Sz = []
            Y_avg_correlation = []
            Y_energy_gap = []
            Y_min_energy = []
            Y_Sz_total = []
            Y_max_chi = []
            Y_avg_entanglement_entropy = []
            ########################################################################

            for L in all_L_values:
                l = L_map[L]

                # print("D: ", D, ", J: ", J, "J_map[J]: ", j, ", L: ", L, ", L_map[L]: ", l, end="")
                parameterValues = all_data[d][j][l]

                ################################################################
                if L == 0 or J == 0:
                    X.append(0)
                else:
                    X.append(L/J)
                Y_avg_Sz.append(abs(parameterValues.ground_state_average_Sz))
                Y_avg_correlation.append(abs(parameterValues.ground_state_average_correlation))
                Y_energy_gap.append(abs(parameterValues.energy_gap))
                Y_min_energy.append(abs(parameterValues.min_energy))
                Y_Sz_total.append(abs(parameterValues.ground_state_Sz_total))
                Y_max_chi.append(abs(parameterValues.ground_state_max_chi))
                Y_avg_entanglement_entropy.append(abs(parameterValues.ground_state_average_entanglement_entropy))
                ################################################################

                all_min_energies[j][l] = parameterValues.min_energy
                all_energy_gaps[j][l] = parameterValues.energy_gap
                all_average_Sz[j][l] = parameterValues.ground_state_average_Sz
                all_total_Sz[j][l] = parameterValues.ground_state_Sz_total
                all_average_correlations[j][l] = parameterValues.ground_state_average_correlation
                all_max_chis[j][l] = parameterValues.ground_state_max_chi
                all_average_entanglement_entropies[j][l] = parameterValues.ground_state_average_entanglement_entropy

                # sys.exit()

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

                # Energies calculated above are computed just for one elementary cell. We have to multiply it by 4,
                # because this script is associated to a 2x2 lattice
                classical_ground_state_energy *= 4
                all_classical_ground_state_energies[j][l] = classical_ground_state_energy
                all_differences_classical_quantum_energies[j][l] = (classical_ground_state_energy - parameterValues.min_energy)
                if not isclose(classical_ground_state_energy, 0.0) and not isclose(parameterValues.min_energy, 0.0):
                    all_differences_classical_quantum_energies[j][l] /= (abs(classical_ground_state_energy) + abs(parameterValues.min_energy))

                # print(" -> ", all_classical_ground_state_phases[j][l])

            ########################################################################
            if J == 0.25:
                X_cross_cut.append(X)
                Y_cross_cut_avg_Sz.append(Y_avg_Sz)
                Y_cross_cut_avg_correlation.append(Y_avg_correlation)

                Y_cross_cut_min_energy.append(Y_min_energy)
                Y_cross_cut_energy_gap.append(Y_energy_gap)
                Y_cross_cut_Sz_total.append(Y_Sz_total)
                Y_cross_cut_max_chi.append(Y_max_chi)
                Y_cross_cut_avg_entanglement_entropy.append(Y_avg_entanglement_entropy)

                Ds_cross_cut.append(D)
                # plot_avg_Sz_vs_lambda_by_J(X, Y, J)
                # sys.exit()
            ########################################################################

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

        if D == D_max:
            visible_sidebar = True
        else:
            visible_sidebar = False

        """
        # Generate maps (version with hard-coded vmin and vax, so that the plots are scaled appropriately)
        plot_2D_map_hard_boundaries(np.asarray(all_min_energies).T, 'viridis', "D = " + str(D), min_J, max_J, "J", min_L, max_L, "$\lambda$", -21.0, 0.0, "D="+str(D)+"_ground_state_energy.pdf", visible_sidebar=visible_sidebar) # Ground state energy
        plot_2D_map_hard_boundaries(np.asarray(all_energy_gaps).T, 'magma', "D = " + str(D), min_J, max_J, "J", min_L, max_L, "$\lambda$", 0.0, 3.65, "D="+str(D)+"_energy_gap.pdf", visible_sidebar=visible_sidebar) # Energy gap
        plot_2D_map_hard_boundaries(np.asarray(all_average_Sz).T, 'inferno', "D = " + str(D), min_J, max_J, "J", min_L, max_L, "$\lambda$", -1.5, 1.5, "D="+str(D)+"_average_Sz.pdf", visible_sidebar=visible_sidebar) # Average value of Sz
        plot_2D_map_hard_boundaries_nonlinear(np.asarray(all_total_Sz).T, 'magma', "D = " + str(D), min_J, max_J, "J", min_L, max_L, "$\lambda$", -12, 12, "D="+str(D)+"_total_Sz.pdf", jump=1, visible_sidebar=visible_sidebar) # Total value of Sz
        plot_2D_map_hard_boundaries(np.asarray(all_average_correlations).T, 'cividis', "D = " + str(D), min_J, max_J, "J", min_L, max_L, "$\lambda$", -2.5, 2.5, "D="+str(D)+"_average_correlation.pdf", visible_sidebar=visible_sidebar) # Average correlation between nodes "in-plane"
        plot_2D_map_hard_boundaries(np.asarray(all_max_chis).T, 'magma', "D = " + str(D), min_J, max_J, "J", min_L, max_L, "$\lambda$", 0, 256, "D="+str(D)+"_max_chi.pdf", visible_sidebar=visible_sidebar) # Max chi
        plot_2D_map_hard_boundaries(np.asarray(all_average_entanglement_entropies).T, 'viridis', "D = " + str(D), min_J, max_J, "J", min_L, max_L, "$\lambda$", 0.0, 1.9, "D="+str(D)+"_average_entanglement_entropy.pdf", visible_sidebar=visible_sidebar) # Average antanglement entropy
        plot_2D_map_classical_results(np.asarray(all_classical_ground_state_phases).T, 'viridis', "D = " + str(D), min_J, max_J, "J", min_L, max_L, "$\lambda$", "D="+str(D)+"_classical_groundstate_phase.pdf", visible_sidebar=visible_sidebar) # Classical groundstate phase
        plot_2D_map_hard_boundaries(np.asarray(all_classical_ground_state_energies).T, 'inferno', "D = " + str(D), min_J, max_J, "J", min_L, max_L, "$\lambda$", -13.0, 0.0, "D="+str(D)+"_classical_groundstate_energy.pdf", visible_sidebar=visible_sidebar) # Classical groundstate energy
        plot_2D_map_hard_boundaries(np.asarray(all_differences_classical_quantum_energies).T, None, "D = " + str(D), min_J, max_J, "J", min_L, max_L, "$\lambda$", -1.0, 1.0, "D="+str(D)+"_classical_quantum_differences.pdf", visible_sidebar=visible_sidebar) # Differences between energies in classical and quantum (DMRG) cases

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
        """

    # print()
    # print("#"*50)
    # print("Ground state energy: [", groundstate_energy_min, " , ", groundstate_energy_max, "]")
    # print("Energy gap: [", energy_gap_min, " , ", energy_gap_max, "]")
    # print("Average Sz: [", average_Sz_min, " , ", average_Sz_max, "]")
    # print("Total Sz: [", total_Sz_min, " , ", total_Sz_max, "]")
    # print("Average correlations: [", average_correlations_min, " , ", average_correlations_max, "]")
    # print("Max chi: [", max_chi_min, " , ", max_chi_max, "]")
    # print("Average entanglement entropy: [", average_entanglement_min, " , ", average_entanglement_max, "]")
    # print("Classical groundstate energy: [", classical_groundstate_energy_min, " , ", classical_groundstate_energy_max, "]")
    # print("Classical-quantum energy differences: [", classical_quantum_differences_min, " , ", classical_quantum_differences_max, "]")
    # print("#"*50)

    J=0.25
    fig = plt.figure(figsize=(10,10))
    ax_avg_Sz = plot_value_vs_lambda_by_J(X_cross_cut, Y_cross_cut_avg_Sz, Ds_cross_cut, J=J, y_label=r'$ |\langle S^Z \rangle | $', target_filename='J='+str(J)+'_avg_Sz.svg', fig=fig, index=1)
    ax_avg_correlation = plot_value_vs_lambda_by_J(X_cross_cut, Y_cross_cut_avg_correlation, Ds_cross_cut, J=J, y_label=r'$ | \langle S^X_i S^Y_j \rangle | $', target_filename='J='+str(J)+'_avg_correlation.svg', fig=fig, index=2)
    ax_min_energy = plot_value_vs_lambda_by_J(X_cross_cut, Y_cross_cut_min_energy, Ds_cross_cut, J=J, y_label=r'$ |E_{MIN}| $', target_filename='J='+str(J)+'_min_energy.svg', fig=fig, index=3)
    ax_energy_gap = plot_value_vs_lambda_by_J(X_cross_cut, Y_cross_cut_energy_gap, Ds_cross_cut, J=J, y_label=r'$ |E_{GAP}| $', target_filename='J='+str(J)+'_energy_gap.svg', fig=fig, index=4)
    # ax_Sz_total = plot_value_vs_lambda_by_J(X_cross_cut, Y_cross_cut_Sz_total, Ds_cross_cut, J=J, y_label=r'$ |S^Z_{TOTAL}| $', target_filename='J='+str(J)+'_Sz_total.svg')
    # ax_max_chi = plot_value_vs_lambda_by_J(X_cross_cut, Y_cross_cut_max_chi, Ds_cross_cut, J=J, y_label=r'$ | \chi_{MAX}| $', target_filename='J='+str(J)+'_max_chi.svg')
    # ax_avg_entanglement_entropy = plot_value_vs_lambda_by_J(X_cross_cut, Y_cross_cut_avg_entanglement_entropy, Ds_cross_cut, J=J, y_label=r'$| \langle \mathcal{S} \rangle | $', target_filename='J='+str(J)+'_avg_enatanglement_entropy.svg')

    # plt.subplot(ax_energy_gap)
    # plt.subplot(ax_min_energy)
    # plt.subplot(ax_avg_Sz)
    # plt.subplot(ax_avg_correlation)
    plt.savefig("panel_plot_2.pdf")
    plt.close()
