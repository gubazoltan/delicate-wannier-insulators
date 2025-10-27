import numpy as np
import sympy as sp
import base_code as bc


def surface_gaps(H_sym, ksymbols, params):
    # unpack the parameters
    Nx, Ny, Nz, Nbands = params["Nx"], params["Ny"], params["Nz"], params["Nbands"]
    kx_sym, ky_sym, kz_sym = ksymbols
    # get the Brillouin zone
    Kxs, Kys, Kzs = bc.get_BZ(params=params)

    # calculate the eigenvalues and eigenstates of the system
    hfunc = sp.lambdify((kx_sym, ky_sym, kz_sym), H_sym, modules="numpy")
    eigenvalues, _ = bc.get_eigsystem(hfunc=hfunc, params=params)
    flattened_bulk = eigenvalues.flatten()

    # calculate the Hamiltonian in the slab geometry
    slab_hfunc_Y = bc.get_surface_hamiltonian(
        h_symbolic=H_sym, ksymbols=ksymbols, params=params, direction="y"
    )
    slab_hfunc_Z = bc.get_surface_hamiltonian(
        h_symbolic=H_sym, ksymbols=ksymbols, params=params, direction="z"
    )

    # calculate the eigenvalues in the slab geometry for the y direction
    eigenvalues = np.zeros((Nx, Nz, Ny * Nbands))
    for i, kx in enumerate(Kxs):
        for k, kz in enumerate(Kzs):
            h = slab_hfunc_Y(kx, kz)
            assert np.allclose(h, h.T.conj()), "Hamiltonian is not hermitian"

            vals = np.linalg.eigvalsh(h)
            vals.sort()
            eigenvalues[i, k, :] = vals
    flattened_Y = eigenvalues.flatten()

    # for the z direction
    eigenvalues = np.zeros((Nx, Ny, Nz * Nbands))
    for i, kx in enumerate(Kxs):
        for j, ky in enumerate(Kys):
            h = slab_hfunc_Z(kx, ky)

            assert np.allclose(h, h.T.conj()), "Hamiltonian is not hermitian"

            vals = np.linalg.eigvalsh(h)
            vals.sort()
            eigenvalues[i, j, :] = vals
    flattened_Z = eigenvalues.flatten()

    return flattened_bulk, flattened_Y, flattened_Z


def wannier_gaps(H_sym, ksymbols, params, nW=False, shift=False):
    # unpack the parameters
    Ny, Nz, Nocc = (
        params["Ny"],
        params["Nz"],
        params["Nocc"],
    )
    kx_sym, ky_sym, kz_sym = ksymbols

    # get the hamiltonian as a function
    hfunc = sp.lambdify((kx_sym, ky_sym, kz_sym), H_sym, modules="numpy")

    # calculate the eigenstates
    _, eigenvectors = bc.get_eigsystem(hfunc=hfunc, params=params, shift=shift)

    # calculate the overlap of the states in the z direction
    links_occ_y, _ = bc.get_links(eigenvectors=eigenvectors, Nocc=Nocc, direction="y")
    links_occ_z, _ = bc.get_links(eigenvectors=eigenvectors, Nocc=Nocc, direction="z")

    # get Wilson loops
    W_occ_y = bc.get_wilsonloops(links=links_occ_y, direction="y")
    W_occ_z = bc.get_wilsonloops(links=links_occ_z, direction="z")

    # get Wilson loop eigenvalues
    nu_vals_occ_y, nu_vecs_occ_y = bc.get_Wilson_eigsystem_schur(Wilsonloops=W_occ_y)
    nu_vals_occ_z, nu_vecs_occ_z = bc.get_Wilson_eigsystem_schur(Wilsonloops=W_occ_z)

    # get the nested wilson loops
    if nW:
        # compute the Wannier basis
        wb_occ_y_lower = bc.get_Wannierbasis(
            eigenvectors=eigenvectors,
            nu_vecs=nu_vecs_occ_y,
            Nocc=Nocc,
            Energy_subspace="occ",
            Wilson_subspace="lower",
        )
        wb_occ_y_upper = bc.get_Wannierbasis(
            eigenvectors=eigenvectors,
            nu_vecs=nu_vecs_occ_y,
            Nocc=Nocc,
            Energy_subspace="occ",
            Wilson_subspace="upper",
        )

        wb_occ_z_lower = bc.get_Wannierbasis(
            eigenvectors=eigenvectors,
            nu_vecs=nu_vecs_occ_z,
            Nocc=Nocc,
            Energy_subspace="occ",
            Wilson_subspace="lower",
        )
        wb_occ_z_upper = bc.get_Wannierbasis(
            eigenvectors=eigenvectors,
            nu_vecs=nu_vecs_occ_z,
            Nocc=Nocc,
            Energy_subspace="occ",
            Wilson_subspace="upper",
        )

        # calculate the nested Wilson loops
        nWl_occ_lower_yz = bc.get_nestedwilson(
            Wannierbasis=wb_occ_y_lower, direction="z"
        )
        nWl_occ_upper_yz = bc.get_nestedwilson(
            Wannierbasis=wb_occ_y_upper, direction="z"
        )
        nWl_occ_lower_zy = bc.get_nestedwilson(
            Wannierbasis=wb_occ_z_lower, direction="y"
        )
        nWl_occ_upper_zy = bc.get_nestedwilson(
            Wannierbasis=wb_occ_z_upper, direction="y"
        )

        # get the eigenvalues of the nested Wilson loops
        nnuvals_occ_lower_yz = bc.get_nestedWilson_eigs(nestedWilson=nWl_occ_lower_yz)
        nnuvals_occ_upper_yz = bc.get_nestedWilson_eigs(nestedWilson=nWl_occ_upper_yz)
        nnuvals_occ_lower_zy = bc.get_nestedWilson_eigs(nestedWilson=nWl_occ_lower_zy)
        nnuvals_occ_upper_zy = bc.get_nestedWilson_eigs(nestedWilson=nWl_occ_upper_zy)

        # make the values continuous
        for j in range(Ny):
            nnuvals_occ_lower_yz[:, j] = bc.make_values_continuous(
                nnuvals_occ_lower_yz[:, j]
            )
            nnuvals_occ_upper_yz[:, j] = bc.make_values_continuous(
                nnuvals_occ_upper_yz[:, j]
            )
        for k in range(Nz):
            nnuvals_occ_lower_zy[:, k] = bc.make_values_continuous(
                nnuvals_occ_lower_zy[:, k]
            )
            nnuvals_occ_upper_zy[:, k] = bc.make_values_continuous(
                nnuvals_occ_upper_zy[:, k]
            )

        return (
            nu_vals_occ_y,
            nu_vals_occ_z,
            wb_occ_y_lower,
            wb_occ_y_upper,
            wb_occ_z_lower,
            wb_occ_z_upper,
            nnuvals_occ_lower_yz,
            nnuvals_occ_upper_yz,
            nnuvals_occ_lower_zy,
            nnuvals_occ_upper_zy,
        )

    else:
        return nu_vals_occ_y, nu_vals_occ_z
