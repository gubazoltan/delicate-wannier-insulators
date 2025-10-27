import sympy as sp
import numpy as np
import gaps
import base_code as bc
import sys

# read in system parameters
Nx = int(sys.argv[1])
Ny = int(sys.argv[2])
Ncoupling = int(sys.argv[3])
alpha_val = float(sys.argv[4])

# define alpha critical
alpha_crit = (np.sqrt(33) - 1) / 8

kx_sym, ky_sym = sp.symbols("kx ky", reals=True)
alpha_sym, beta_sym, gamma_z, lambda_z = sp.symbols(
    "alpha beta gamma_z lambda_z", reals=True
)
ksymbols = [kx_sym, ky_sym]
symbols = [alpha_sym, beta_sym, lambda_z, gamma_z]

H_layered = bc.get_layeredHamiltonian(ksymbols, symbols)

# define system size
Nbands = 4
Nocc = 2

params = {}
params["Nx"] = Nx
params["Ny"] = Ny
params["Nbands"] = Nbands
params["Nocc"] = Nocc

# create mesh for phase diagram
gammas = np.linspace(0.5, 1.5, Ncoupling, endpoint=True)

# make container for the extracted data
gap_data = np.zeros((Ncoupling, 9))

for j in range(Ncoupling):
    # print the parameters
    print("alpha: ", alpha_val, "\t" "gamma: ", gammas[j])

    # save the alpha and gamma values for the point in the phase diagram
    gap_data[j, 0], gap_data[j, 1] = alpha_val, gammas[j]

    # fix the parameters of the model
    H_fixedparam = H_layered.subs(
        {
            alpha_sym: alpha_val * alpha_crit,
            beta_sym: 2.0,
            gamma_z: gammas[j],
            lambda_z: 1,
        }
    )

    gap_data[j, 2:5] = gaps.energy_gaps(
        H_sym=H_fixedparam, ksymbols=ksymbols, params=params
    )
    gap_data[j, 5:9] = gaps.wannier_gaps(
        H_sym=H_fixedparam, ksymbols=ksymbols, params=params, shift=True
    )

    # print the gaps
    print(
        "bulk gap: ",
        gap_data[j, 2],
        "\n" + "x gap: ",
        gap_data[j, 3],
        "\n" + "y gap: ",
        gap_data[j, 4],
    )

    print(
        "Wannier x gap 0: ",
        gap_data[j, 5],
        "\n" + "Wannier x gap 1/2: ",
        gap_data[j, 6],
        "\n" + "Wannier y gap 0: ",
        gap_data[j, 7],
        "\n" + "Wannier y gap 1/2: ",
        gap_data[j, 8],
    )

# define filename for the data
filename = "pd_" + f"{alpha_val:.6f}_{Ncoupling}_{Nx}_{Ny}" + ".txt"
np.savetxt(fname=filename, X=gap_data)
