import sympy as sp
import numpy as np
import gaps
import base_code as bc
import sys

# read in system parameters
Nx = int(sys.argv[1])
Ny = int(sys.argv[2])
Nz = int(sys.argv[3])
Ncoupling = int(sys.argv[4])
alpha_val = float(sys.argv[5])

# create parameters
kx_sym, ky_sym, kz_sym = sp.symbols("k_x k_y k_z", real=True)
ksymbols = [kx_sym, ky_sym, kz_sym]
alpha = sp.symbols("alpha", real=True, positive=True)
gamma_z, lambda_z = sp.symbols("gamma_z lambda_z", real=True)

# create the layered RTP Hamiltonian
H_layered = bc.create_layeredRTP(ksymbols, [alpha, gamma_z, lambda_z])

# define system parameters
Nbands = 4
Nocc = 2

params = {}
params["Nx"] = Nx
params["Ny"] = Ny
params["Nz"] = Nz
params["Nbands"] = Nbands
params["Nocc"] = Nocc

# create mesh for phase diagram
gammas = np.linspace(0.5, 1.5, Ncoupling, endpoint=True)

# make container for the extracted data
gap_data = np.zeros((Ncoupling, 9))

# obtain gap data for the phase diagram
for j in range(Ncoupling):
    # print the parameters
    print("alpha: ", alpha_val, "\t" "gamma: ", gammas[j])

    # save the alpha and gamma values for the point in the phase diagram
    gap_data[j, 0], gap_data[j, 1] = alpha_val, gammas[j]

    # fix the parameters of the model
    H_fixedparam = H_layered.subs({alpha: alpha_val, gamma_z: gammas[j], lambda_z: 1})

    # compute the bulk and surface energy gaps
    eigenvalues_flattened_bulk, eigenvalues_flattened_Y, eigenvalues_flattened_Z = (
        gaps.surface_gaps(H_sym=H_fixedparam, ksymbols=ksymbols, params=params)
    )

    # sort the data and compute the gaps
    eigenvalues_flattened_bulk.sort()
    eigenvalues_flattened_Z.sort()
    eigenvalues_flattened_Y.sort()

    bulk_gap = (
        eigenvalues_flattened_bulk[Nx * Ny * Nz * 2]
        - eigenvalues_flattened_bulk[Nx * Ny * Nz * 2 - 1]
    )
    surfy_gap = (
        eigenvalues_flattened_Y[Nx * Ny * Nz * 2]
        - eigenvalues_flattened_Y[Nx * Ny * Nz * 2 - 1]
    )
    surfz_gap = (
        eigenvalues_flattened_Z[Nx * Ny * Nz * 2]
        - eigenvalues_flattened_Z[Nx * Ny * Nz * 2 - 1]
    )

    # print data
    print(
        "Bulk gap: ",
        bulk_gap,
        "\n" + "Surface y gap: ",
        surfy_gap,
        "\n" + "Surface z gap: ",
        surfz_gap,
    )

    # save bulk and surface energy gap
    gap_data[j, 2] = bulk_gap
    gap_data[j, 3] = surfy_gap
    gap_data[j, 4] = surfz_gap

    # compute the Wannier gap data
    wannier_data = gaps.wannier_gaps(
        H_sym=H_fixedparam, ksymbols=ksymbols, params=params, shift=True
    )

    # extract the Wannier bands
    nu_vals_occ_y = wannier_data[0]
    nu_vals_occ_z = wannier_data[1]

    # check the gap between the Wannier bands both around 0 and 1/2
    wannier_y_gap0 = np.min(abs(nu_vals_occ_y[:, :, :, 1] - nu_vals_occ_y[:, :, :, 0]))
    wannier_y_gap1 = np.min(
        abs(1 + nu_vals_occ_y[:, :, :, 0] - nu_vals_occ_y[:, :, :, 1])
    )
    wannier_z_gap0 = np.min(abs(nu_vals_occ_z[:, :, :, 1] - nu_vals_occ_z[:, :, :, 0]))
    wannier_z_gap1 = np.min(
        abs(1 + nu_vals_occ_z[:, :, :, 0] - nu_vals_occ_z[:, :, :, 1])
    )

    # save the Wannier gap data
    gap_data[j, 5] = wannier_y_gap0
    gap_data[j, 6] = wannier_y_gap1
    gap_data[j, 7] = wannier_z_gap0
    gap_data[j, 8] = wannier_z_gap1

    # print the Wannier gap data
    print(
        "Wannier y gap 0: ",
        wannier_y_gap0,
        "\n" + "Wannier y gap 1/2: ",
        wannier_y_gap1,
        "\n" + "Wannier z gap 0: ",
        wannier_z_gap0,
        "\n" + "Wannier z gap 1/2: ",
        wannier_z_gap1,
        "\n",
    )

# define filename for the data container
filename = "phase_diagram_" + f"{alpha_val:.6f}_{Ncoupling}_{Nx}_{Ny}_{Nz}" + ".txt"
np.savetxt(fname=filename, X=gap_data)
