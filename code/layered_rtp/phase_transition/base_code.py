import numpy as np
import scipy as scp
import sympy as sp
from sympy.physics.quantum import TensorProduct


def create_layeredRTP(ksymbols, symbs):
    # unpack the symbols
    kx_sym, ky_sym, kz_sym = ksymbols
    alpha, gamma_z, lambda_z = symbs

    # create the Pauli matrices
    s0 = sp.eye(2)
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])

    # create RTP Hamiltonian
    hrtp = sp.sin(2 * kx_sym) * sx
    hrtp += sp.sin(kx_sym) * sp.sin(ky_sym) * sy
    hrtp += -(alpha + sp.cos(2 * kx_sym) + sp.cos(ky_sym)) * sz
    # create layered RTP Hamiltonian
    H_layered = sp.Matrix(np.zeros((4, 4)))
    H_layered += TensorProduct(sz, hrtp)
    H_layered += TensorProduct(sy, s0) * lambda_z * sp.sin(kz_sym)
    H_layered += TensorProduct(sx, s0) * (gamma_z + lambda_z * sp.cos(kz_sym))

    return H_layered


def make_values_continuous(nnu_vals):
    # make the values continuous
    if len(nnu_vals.shape) == 2:
        (N1, N2) = nnu_vals.shape

        for k in range(N2):
            for i in range(1, N1):
                diff = nnu_vals[i, k] - nnu_vals[i - 1, k]
                if abs(diff) > 0.5:
                    nnu_vals[i, k] -= 1 * np.sign(diff)
    else:
        N1 = len(nnu_vals)
        for i in range(1, N1):
            diff = nnu_vals[i] - nnu_vals[i - 1]
            if abs(diff) > 0.5:
                nnu_vals[i] -= 1 * np.sign(diff)

    return nnu_vals


def get_BZ(params, shift=False):
    # extract the parameters
    Nx, Ny, Nz = params["Nx"], params["Ny"], params["Nz"]

    # define the brillouin zone
    Kxs = np.linspace(-np.pi, np.pi, Nx, endpoint=False)
    Kys = np.linspace(-np.pi, np.pi, Ny, endpoint=False)
    Kzs = np.linspace(-np.pi, np.pi, Nz, endpoint=False)

    # shift the BZ to avoid high symmetry points
    if shift:
        Kxs += 1e-10
        Kys += 1e-10
        Kzs += 1e-10

    # return the brillouin zone
    return Kxs, Kys, Kzs


def get_eigsystem(hfunc, params, shift=False):
    # extract the parameters
    Nx, Ny, Nz, Nbands = params["Nx"], params["Ny"], params["Nz"], params["Nbands"]

    # define the brillouin zone
    Kxs, Kys, Kzs = get_BZ(params=params, shift=shift)

    # define container for eigenvalues and eigenvectors
    eigenvalues = np.zeros((Nx, Ny, Nz, Nbands))
    eigenvectors = np.zeros((Nx, Ny, Nz, Nbands, Nbands)).astype(np.complex128)

    # loop over the brillouin zone
    for i, kx in enumerate(Kxs):
        for j, ky in enumerate(Kys):
            for k, kz in enumerate(Kzs):
                # calculate the eigenvalues and eigenvectors
                evals, evecs = np.linalg.eigh(hfunc(kx, ky, kz))

                # sort the eigenvalues and eigenvectors
                idx = np.argsort(evals)
                evals = evals[idx]
                evecs = evecs[:, idx]

                # store the eigenvalues and eigenvectors
                eigenvalues[i, j, k, :] = evals[:]
                eigenvectors[i, j, k, :, :] = evecs[:, :]

    # return the eigenvalues and eigenvectors
    return eigenvalues, eigenvectors


def get_links(eigenvectors, Nocc, direction="x"):
    # get the parameters of the system
    (Nx, Ny, Nz, Nbands, _) = eigenvectors.shape

    # define container for the links
    links_occ = np.zeros((Nx, Ny, Nz, Nocc, Nocc)).astype(np.complex128)
    links_unocc = np.zeros((Nx, Ny, Nz, Nbands - Nocc, Nbands - Nocc)).astype(
        np.complex128
    )

    if direction == "x":
        # loop over the brillouin zone
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    # get the overlap matrix
                    overlap_occ = (
                        eigenvectors[(i + 1) % Nx, j, k, :, :Nocc].conj().T
                        @ eigenvectors[i, j, k, :, :Nocc]
                    )
                    overlap_unocc = (
                        eigenvectors[(i + 1) % Nx, j, k, :, Nocc:].conj().T
                        @ eigenvectors[i, j, k, :, Nocc:]
                    )

                    # do the singular value decomposition
                    U_occ, _, Vh_occ = np.linalg.svd(overlap_occ)
                    U_unocc, _, Vh_unocc = np.linalg.svd(overlap_unocc)

                    # store the links
                    links_occ[i, j, k, :, :] = U_occ @ Vh_occ
                    links_unocc[i, j, k, :, :] = U_unocc @ Vh_unocc

    elif direction == "y":
        # loop over the brillouin zone
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    # get the overlap matrix
                    overlap_occ = (
                        eigenvectors[i, (j + 1) % Ny, k, :, :Nocc].conj().T
                        @ eigenvectors[i, j, k, :, :Nocc]
                    )
                    overlap_unocc = (
                        eigenvectors[i, (j + 1) % Ny, k, :, Nocc:].conj().T
                        @ eigenvectors[i, j, k, :, Nocc:]
                    )

                    # do the singular value decomposition
                    U_occ, _, Vh_occ = np.linalg.svd(overlap_occ)
                    U_unocc, _, Vh_unocc = np.linalg.svd(overlap_unocc)

                    # store the links
                    links_occ[i, j, k, :, :] = U_occ @ Vh_occ
                    links_unocc[i, j, k, :, :] = U_unocc @ Vh_unocc

    elif direction == "z":
        # loop over the brillouin zone
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    # get the overlap matrix
                    overlap_occ = (
                        eigenvectors[i, j, (k + 1) % Nz, :, :Nocc].conj().T
                        @ eigenvectors[i, j, k, :, :Nocc]
                    )
                    overlap_unocc = (
                        eigenvectors[i, j, (k + 1) % Nz, :, Nocc:].conj().T
                        @ eigenvectors[i, j, k, :, Nocc:]
                    )

                    # do the singular value decomposition
                    U_occ, _, Vh_occ = np.linalg.svd(overlap_occ)
                    U_unocc, _, Vh_unocc = np.linalg.svd(overlap_unocc)

                    # store the links
                    links_occ[i, j, k, :, :] = U_occ @ Vh_occ
                    links_unocc[i, j, k, :, :] = U_unocc @ Vh_unocc

    else:
        raise ValueError("Invalid direction")

    # return the links
    return links_occ, links_unocc


def get_wilsonloops(links, direction="x"):
    # get the parameters of the system
    (Nx, Ny, Nz, Noccunocc, _) = links.shape

    # define container for the wilson loops
    Wilsonloops = np.zeros((Nx, Ny, Nz, Noccunocc, Noccunocc)).astype(np.complex128)

    if direction == "x":
        # loop over the brillouin zone
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):

                    # define container for the Wilson loop
                    W = np.eye(Noccunocc).astype(np.complex128)

                    # iterate over the x direction
                    for kp in range(Nx):
                        W = links[(i + kp) % Nx, j, k, :, :] @ W

                    # store the Wilson loop
                    Wilsonloops[i, j, k, :, :] = W

    elif direction == "y":
        # loop over the brillouin zone
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):

                    # define container for the Wilson loop
                    W = np.eye(Noccunocc).astype(np.complex128)

                    # iterate over the y direction
                    for kp in range(Ny):
                        W = links[i, (j + kp) % Ny, k, :, :] @ W

                    # store the Wilson loop
                    Wilsonloops[i, j, k, :, :] = W

    # this works only if the Wilson loop is computed in the occupied subspace
    elif direction == "z":
        # loop over the brillouin zone
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):

                    # define container for the Wilson loop
                    W = np.eye(Noccunocc).astype(np.complex128)

                    # iterate over the z direction
                    for kp in range(Nz):
                        W = links[i, j, (k + kp) % Nz, :, :] @ W

                    # store the Wilson loop
                    Wilsonloops[i, j, k, :, :] = W
    else:
        raise ValueError("Invalid direction")

    # return the Wilson loops
    return Wilsonloops


def get_Wilson_eigsystem(Wilsonloops):
    # get the parameters of the system
    (Nx, Ny, Nz, Noccunocc, _) = Wilsonloops.shape

    # define container for the eigenvalues and eigenvectors
    nu_vals = np.zeros((Nx, Ny, Nz, Noccunocc))
    nu_vecs = np.zeros((Nx, Ny, Nz, Noccunocc, Noccunocc)).astype(np.complex128)

    # loop over the brillouin zone
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # calculate the Wilson loop eigenvalues and eigenvectors
                evals, evecs = np.linalg.eig(Wilsonloops[i, j, k, :, :])
                angles = np.angle(evals) / (2 * np.pi)

                # sort the eigenvalues and eigenvectors
                idx = np.argsort(angles)
                angles = angles[idx]
                evecs = evecs[:, idx]

                # store the eigenvalues and eigenvectors
                nu_vals[i, j, k, :] = angles[:]
                nu_vecs[i, j, k, :, :] = evecs[:, :]

    # return the eigenvalues and eigenvectors
    return nu_vals, nu_vecs


def get_Wilson_eigsystem_schur(Wilsonloops):
    # get the parameters of the system
    (Nx, Ny, Nz, Noccunocc, _) = Wilsonloops.shape

    # define container for the eigenvalues and eigenvectors
    nu_vals = np.zeros((Nx, Ny, Nz, Noccunocc))
    nu_vecs = np.zeros((Nx, Ny, Nz, Noccunocc, Noccunocc)).astype(np.complex128)

    # loop over the brillouin zone
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # calculate the Wilson loop eigenvalues and eigenvectors
                T, Z = scp.linalg.schur(Wilsonloops[i, j, k, :, :])
                evals = scp.linalg.eigvals(T)
                angles = np.angle(evals) / (2 * np.pi)

                # sort the eigenvalues and eigenvectors
                idx = np.argsort(angles)
                angles = angles[idx]
                evecs = Z[:, idx]

                # store the eigenvalues and eigenvectors
                nu_vals[i, j, k, :] = angles[:]
                nu_vecs[i, j, k, :, :] = evecs[:, :]

    # return the eigenvalues and eigenvectors
    return nu_vals, nu_vecs


def get_Wannierbasis(
    eigenvectors, nu_vecs, Nocc, Energy_subspace="occ", Wilson_subspace="lower"
):
    # get the parameters of the system
    (Nx, Ny, Nz, Nbands, _) = eigenvectors.shape

    # check the Energy subspace
    if Energy_subspace == "occ":
        E_ind = 0
    elif Energy_subspace == "unocc":
        E_ind = Nocc
    else:
        raise ValueError("Invalid Energy subspace")

    # check the Wilson subspace
    if Wilson_subspace == "lower":
        W_ind = 0
    elif Wilson_subspace == "upper":
        W_ind = 1
    else:
        raise ValueError("Invalid subspace")

    # define container for the Wannier basis
    Wannierbasis = np.zeros((Nx, Ny, Nz, Nbands)).astype(np.complex128)

    # loop over the brillouin zone
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # calculate the Wannier basis
                Wannierbasis[i, j, k, :] = (
                    eigenvectors[i, j, k, :, E_ind + 0] * nu_vecs[i, j, k, 0, W_ind]
                    + eigenvectors[i, j, k, :, E_ind + 1] * nu_vecs[i, j, k, 1, W_ind]
                )

    # return the Wannier basis
    return Wannierbasis


def get_nestedwilson(Wannierbasis, direction):
    # get the parameters of the system
    (Nx, Ny, Nz, _) = Wannierbasis.shape

    if direction == "x":
        # define container for the nested Wilson loop
        nestedWilson = np.zeros((Ny, Nz)).astype(np.complex128)

        for j in range(Ny):
            for k in range(Nz):
                # define container for the nestedWilson loop
                W = 1.0

                for i in range(Nx):
                    # obtain the overlap matrix
                    overlap = (
                        Wannierbasis[(i + 1) % Nx, j, k, :].conj().T
                        @ Wannierbasis[i, j, k, :]
                    )

                    # store the nestedWilson loop
                    W = overlap * W / np.abs(overlap)

                # store the nestedWilson loop
                nestedWilson[j, k] = W

    elif direction == "y":
        # define container for the nested Wilson loop
        nestedWilson = np.zeros((Nx, Nz)).astype(np.complex128)

        for i in range(Nx):
            for k in range(Nz):
                # define container for the nestedWilson loop
                W = 1.0

                for j in range(Ny):
                    # obtain the overlap matrix
                    overlap = (
                        Wannierbasis[i, (j + 1) % Ny, k, :].conj().T
                        @ Wannierbasis[i, j, k, :]
                    )

                    # store the nestedWilson loop
                    W = overlap * W / np.abs(overlap)

                # store the nestedWilson loop
                nestedWilson[i, k] = W

    elif direction == "z":
        # define container for the nested Wilson loop
        nestedWilson = np.zeros((Nx, Ny)).astype(np.complex128)

        for i in range(Nx):
            for j in range(Ny):
                # define container for the nestedWilson loop
                W = 1.0

                for k in range(Nz):
                    # obtain the overlap matrix
                    overlap = (
                        Wannierbasis[i, j, (k + 1) % Nz, :].conj().T
                        @ Wannierbasis[i, j, k, :]
                    )

                    # store the nestedWilson loop
                    W = overlap * W / np.abs(overlap)

                # store the nestedWilson loop
                nestedWilson[i, j] = W

    else:
        raise ValueError("Invalid direction")

    # return the nestedWilson loop
    return nestedWilson


def get_nestedWilson_eigs(nestedWilson):
    # get the parameters of the system
    (N1, N2) = nestedWilson.shape

    # define container for the eigenvalues and eigenvectors
    nnu_vals = np.zeros((N1, N2))

    # loop over the brillouin zone
    for i in range(N1):
        for j in range(N2):
            # get the phases of the eigenvalues
            nnu_vals[i, j] = np.angle(nestedWilson[i, j]) / (2 * np.pi)

    # return the eigenvalues
    return nnu_vals


def get_surface_hamiltonian(h_symbolic, ksymbols, params, direction="y"):
    # expand the parameters
    Ny, Nz, Nbands = params["Ny"], params["Nz"], params["Nbands"]

    # expand the k symbols
    kx_sym, ky_sym, kz_sym = ksymbols

    if direction == "y":
        # obtain the hopping terms
        Ky_pos = sp.integrate(
            h_symbolic * sp.exp(sp.I * ky_sym * (-1.0)), (ky_sym, -sp.pi, sp.pi)
        ) / (2 * sp.pi)
        Ky_neg = sp.integrate(
            h_symbolic * sp.exp(sp.I * ky_sym), (ky_sym, -sp.pi, sp.pi)
        ) / (2 * sp.pi)
        Ky_pos = Ky_pos.rewrite(sp.cos).simplify()
        Ky_neg = Ky_neg.rewrite(sp.cos).simplify()

        # the diagonal term is
        H_diag = h_symbolic - (
            Ky_pos * sp.exp(sp.I * ky_sym) + Ky_neg * sp.exp(-sp.I * ky_sym)
        )
        H_diag = H_diag.rewrite(sp.cos).simplify()
        H_diag = sp.nsimplify(H_diag, tolerance=1e-8)

        # create container for the Hamiltonian
        h = sp.zeros(Ny * Nbands, Ny * Nbands)

        # iterate over the cells
        for j in range(Ny):
            # add the diagonal term
            h[j * Nbands : (j + 1) * Nbands, j * Nbands : (j + 1) * Nbands] = H_diag[
                :, :
            ]

            # if downwards hopping is possible
            if j > 0:
                h[(j - 1) * Nbands : j * Nbands, j * Nbands : (j + 1) * Nbands] = (
                    Ky_pos[:, :]
                )

            # if upwards hopping is possible
            if j < Ny - 1:
                h[
                    (j + 1) * Nbands : (j + 2) * Nbands, j * Nbands : (j + 1) * Nbands
                ] = Ky_neg[:, :]

        slab_hfunc = sp.lambdify((kx_sym, kz_sym), h, modules="numpy")

    elif direction == "z":
        # obtain the hopping terms
        Kz_pos = sp.integrate(
            h_symbolic * sp.exp(sp.I * kz_sym * (-1.0)), (kz_sym, -sp.pi, sp.pi)
        ) / (2 * sp.pi)
        Kz_neg = sp.integrate(
            h_symbolic * sp.exp(sp.I * kz_sym), (kz_sym, -sp.pi, sp.pi)
        ) / (2 * sp.pi)
        Kz_pos = Kz_pos.rewrite(sp.cos).simplify()
        Kz_neg = Kz_neg.rewrite(sp.cos).simplify()

        # the diagonal term is
        H_diag = h_symbolic - (
            Kz_pos * sp.exp(sp.I * kz_sym) + Kz_neg * sp.exp(-sp.I * kz_sym)
        )
        H_diag = H_diag.rewrite(sp.cos).simplify()
        H_diag = sp.nsimplify(H_diag, tolerance=1e-8)

        # create container for the Hamiltonian
        h = sp.zeros(Nz * Nbands, Nz * Nbands)

        # iterate over the cells
        for k in range(Nz):
            # add the diagonal term
            h[k * Nbands : (k + 1) * Nbands, k * Nbands : (k + 1) * Nbands] = H_diag[
                :, :
            ]

            # if downwards hopping is possible
            if k > 0:
                h[(k - 1) * Nbands : k * Nbands, k * Nbands : (k + 1) * Nbands] = (
                    Kz_pos[:, :]
                )

            # if upwards hopping is possible
            if k < Nz - 1:
                h[
                    (k + 1) * Nbands : (k + 2) * Nbands, k * Nbands : (k + 1) * Nbands
                ] = Kz_neg[:, :]

        slab_hfunc = sp.lambdify((kx_sym, ky_sym), h, modules="numpy")

    else:
        raise ValueError("Invalid direction")

    # return the hamiltonian function
    return slab_hfunc


def get_wire_hamiltonian(h_symbolic, syms, params, corner_pert=None):
    # extract system parameters
    Ny, Nz, Nbands = params["Ny"], params["Nz"], params["Nbands"]

    # create index array
    indf = np.zeros((Ny, Nz), dtype=int)
    for j in range(Ny):
        for k in range(Nz):
            indf[j, k] = j * Nz * Nbands + k * Nbands

    # extract symbolic variables
    kx_sym, ky_sym, kz_sym = syms

    Ly_nn_pos = sp.integrate(
        h_symbolic * sp.exp(sp.I * ky_sym * (-1.0)), (ky_sym, -sp.pi, sp.pi)
    ) / (2 * sp.pi)
    Ly_nn_neg = sp.integrate(
        h_symbolic * sp.exp(sp.I * ky_sym), (ky_sym, -sp.pi, sp.pi)
    ) / (2 * sp.pi)

    Lz_nn_pos = sp.integrate(
        h_symbolic * sp.exp(sp.I * kz_sym * (-1.0)), (kz_sym, -sp.pi, sp.pi)
    ) / (2 * sp.pi)
    Lz_nn_neg = sp.integrate(
        h_symbolic * sp.exp(sp.I * kz_sym), (kz_sym, -sp.pi, sp.pi)
    ) / (2 * sp.pi)

    Ly_nn_pos = Ly_nn_pos.rewrite(sp.cos).simplify()
    Ly_nn_neg = Ly_nn_neg.rewrite(sp.cos).simplify()
    Lz_nn_pos = Lz_nn_pos.rewrite(sp.cos).simplify()
    Lz_nn_neg = Lz_nn_neg.rewrite(sp.cos).simplify()

    H_diag = h_symbolic
    H_diag -= Ly_nn_pos * sp.exp(sp.I * ky_sym) + Ly_nn_neg * sp.exp(-sp.I * ky_sym)
    H_diag -= Lz_nn_pos * sp.exp(sp.I * kz_sym) + Lz_nn_neg * sp.exp(-sp.I * kz_sym)
    H_diag = H_diag.rewrite(sp.cos).simplify()
    H_diag = sp.nsimplify(H_diag, tolerance=1e-8)

    # create container for the Hamiltonian
    h = sp.zeros(Ny * Nz * Nbands, Ny * Nz * Nbands)

    for j in range(Ny):
        for k in range(Nz):
            h[indf[j, k] : indf[j, k + 1], indf[j, k] : indf[j, k + 1]] = H_diag

            if j > 0:
                h[indf[j - 1, k] : indf[j - 1, k + 1], indf[j, k] : indf[j, k + 1]] = (
                    Ly_nn_pos
                )

            if j < Ny - 1:
                h[indf[j + 1, k] : indf[j + 1, k + 1], indf[j, k] : indf[j, k + 1]] = (
                    Ly_nn_neg
                )

            if k > 0:
                h[indf[j, k - 1] : indf[j, k], indf[j, k] : indf[j, k + 1]] = Lz_nn_pos

            if k < Nz - 1:
                h[indf[j, k + 1] : indf[j, k + 2], indf[j, k] : indf[j, k + 1]] = (
                    Lz_nn_neg
                )

    # perturb the corner of the wire
    if corner_pert == "detach":
        # perturb the corner of the wire
        h[indf[Ny - 1, 0] : indf[Ny - 1, 1], indf[Ny - 1, 0] : indf[Ny - 1, 1]] *= 0.2

    # make a function out of the hamiltonian
    wire_hfunc = sp.lambdify(kx_sym, h, modules="numpy")

    # return the hamiltonian function
    return wire_hfunc
