import tenpy.linalg.np_conserved as npc
from tenpy.networks.site import SpinSite
from tenpy.models.lattice import Honeycomb
from tenpy.models.spins import SpinModel
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
import matplotlib.pyplot as plt
import numpy as np
import sys

def HoneyComb_lattice_model(lattice, Jx, Jy, Jz, D, hx=0., hy=0., hz=0., muJ=0., E=0., bcx="periodic", bcy="ladder"):

    model_params = {
        # "S": 1.5,  # Spin 3/2
        # "lattice": "Honeycomb",
        "lattice": lattice,
        "bc_MPS": "finite",
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

def save_lattice(lattice, Lx, Ly):
    ax = plt.gca()
    lattice.plot_coupling(ax, linewidth=3.)
    lattice.plot_order(ax=ax, linestyle=':')
    lattice.plot_sites(ax=ax)
    lattice.plot_basis(ax, origin=-0.5*(lattice.basis[0] + lattice.basis[1]))
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # plt.show()
    filename = "lattice_" + str(Lx) + "_" + str(Ly) + ".pdf"
    plt.savefig(filename)

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Incorrect number of arguments. Provide them in the following form:")
        print("- Lx,")
        print("- Ly,")
        print("- D,")
        print("- J_min,")
        print("- J_max,")
        print("- L_min,")
        print("- L_max.")
        sys.exit()

    Lx = int(sys.argv[1])
    Ly = int(sys.argv[2])
    D = float(sys.argv[3])
    J_min = float(sys.argv[4])
    J_max = float(sys.argv[5])
    L_min = float(sys.argv[6])
    L_max = float(sys.argv[7])
    print("J_min: ", J_min)
    print("J_max: ", J_max)
    print("L_min: ", L_min)
    print("L_max: ", L_max)

    spinSite = SpinSite(S=1.5, conserve='Sz')
    lattice = Honeycomb(Lx=Lx, Ly=Ly, sites=spinSite, bc=['periodic', 'open'])
    save_lattice(lattice, Lx, Ly)
    filename = "parameters_Lx=" + str(Lx) + "_Ly=" + str(Ly) + "_D=" + str(D) + ".csv"
    file = open(filename,"w+")

    # Note, that below we are passing negative values of the above parameter, because
    # the defnition of Hamiltonian in SpinModel is different from the Hamiltonian defined
    # in the article, that we are basing on.
    # bcy can take values: "cylinder" / "ladder"
    # bcx can take values "periodic" / "open"
    model = HoneyComb_lattice_model(lattice=lattice, Jx=-1, Jy=-1, Jz=-1, D=-1, bcx="periodic", bcy="ladder")

    N_sites = model.lat.N_sites
    sectors = []
    for i in range(3*N_sites+1):
        s = int(((-1.5)*N_sites)+i)
        sectors.append(s)
    parameters = ""
    j_range = np.arange(J_max, J_min - 0.05, -0.05)
    l_range = np.arange(L_max, L_min - 0.05, -0.05)

    if j_range[-1] < J_min:
        j_range = j_range[:-1]
    if l_range[-1] < L_min:
        l_range = l_range[:-1]

    for j in j_range:
        for l in l_range:
            for sector in sectors:
                parameters += str(Lx) + " " + str(Ly) + " " + str(round(D, 3)) + " " + str(round(j, 3)) + " " + str(round(l, 3)) + " " + str(round(sector, 3)) + "\n"

    file.write(parameters)
    file.close()
