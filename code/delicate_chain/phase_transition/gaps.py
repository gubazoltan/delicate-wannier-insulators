import numpy as np
import sympy as sp
import base_code as bc


def bulk_gap(hfunc, params):
    # unpack the parameters
    Nx, Ny, Nocc = params["Nx"], params["Ny"], params["Nocc"]

    eigenvalues, _ = bc.get_eigsystem(hfunc=hfunc, params=params)
    flattened_bulk = eigenvalues.flatten()
    flattened_bulk.sort()

    # compute the gap
    gap_bulk = flattened_bulk[Nx * Ny * Nocc] - flattened_bulk[Nx * Ny * Nocc - 1]

    # return the bulk gap
    return gap_bulk


def surface_gap(hfunc, params, direction="x"):
    # unpack the parameters
    Nx, Ny, Nbands, Nocc = params["Nx"], params["Ny"], params["Nbands"], params["Nocc"]

    # compute the BZ
    Kxs, Kys = bc.get_BZ(params=params)

    if direction == "x":
        # Calculate the eigenvalues in the slab geometry for the x direction
        eigenvalues = np.zeros((Ny, Nx * Nbands))
        for j, ky in enumerate(Kys):
            h = hfunc(ky)
            assert np.allclose(h, h.T.conj()), "Hamiltonian is not hermitian"

            vals = np.linalg.eigvalsh(h)
            vals.sort()
            eigenvalues[j, :] = vals
        flattened_X = eigenvalues.flatten()
        flattened_X.sort()

        # compute the gap
        gap_x = flattened_X[Nx * Ny * Nocc] - flattened_X[Nx * Ny * Nocc - 1]

        return gap_x

    elif direction == "y":
        # Calculate the eigenvalues in the slab geometry for the y direction
        eigenvalues = np.zeros((Nx, Ny * Nbands))
        for i, kx in enumerate(Kxs):
            h = hfunc(kx)
            assert np.allclose(h, h.T.conj()), "Hamiltonian is not hermitian"

            vals = np.linalg.eigvalsh(h)
            vals.sort()
            eigenvalues[i, :] = vals
        flattened_Y = eigenvalues.flatten()
        flattened_Y.sort()

        # compute the gap
        gap_y = flattened_Y[Nx * Ny * Nocc] - flattened_Y[Nx * Ny * Nocc - 1]

        return gap_y

    else:
        raise ValueError("Direction must be 'x' or 'y'")


def energy_gaps(H_sym, ksymbols, params):
    kx_sym, ky_sym = ksymbols

    # Calculate the eigenvalues and eigenstates of the system
    hfunc = sp.lambdify((kx_sym, ky_sym), H_sym, modules="numpy")

    # calculate the bulk energy gap
    gap_bulk = bulk_gap(hfunc=hfunc, params=params)

    # Calculate the Hamiltonian in the slab geometry
    slab_hfunc_X = bc.slab_geometry(
        h_symbolic=H_sym, ksymbols=ksymbols, params=params, direction="x"
    )
    gap_x = surface_gap(hfunc=slab_hfunc_X, params=params, direction="x")

    # Calculate the Hamiltonian in the slab geometry
    slab_hfunc_Y = bc.slab_geometry(
        h_symbolic=H_sym, ksymbols=ksymbols, params=params, direction="y"
    )
    gap_y = surface_gap(hfunc=slab_hfunc_Y, params=params, direction="y")

    return [gap_bulk, gap_x, gap_y]


def wannier_gaps(H_sym, ksymbols, params, shift=False):
    Nocc = params["Nocc"]
    kx_sym, ky_sym = ksymbols

    # get the hamiltonian as a function
    hfunc = sp.lambdify((kx_sym, ky_sym), H_sym, modules="numpy")

    # calculate the eigenstates
    _, eigenvectors = bc.get_eigsystem(hfunc=hfunc, params=params, shift=shift)

    # calculate the overlap of the states
    links_x, _ = bc.get_links(eigenvectors=eigenvectors, Nocc=Nocc, direction="x")
    links_y, _ = bc.get_links(eigenvectors=eigenvectors, Nocc=Nocc, direction="y")

    # calculate wilson loops
    Wx = bc.get_wilsonloops(links=links_x, direction="x")
    Wy = bc.get_wilsonloops(links=links_y, direction="y")

    nu_x, _ = bc.get_Wilson_eigsystem(Wilsonloops=Wx)
    nu_y, _ = bc.get_Wilson_eigsystem(Wilsonloops=Wy)

    gapx0 = np.min(abs(nu_x[:, :, 1] - nu_x[:, :, 0]))
    gapx12 = np.min(abs(1 + nu_x[:, :, 0] - nu_x[:, :, 1]))
    gapy0 = np.min(abs(nu_y[:, :, 1] - nu_y[:, :, 0]))
    gapy12 = np.min(abs(1 + nu_y[:, :, 0] - nu_y[:, :, 1]))

    return [gapx0, gapx12, gapy0, gapy12]


def wannier_gaps_functional(hfunc, params, shift=False):
    Nocc = params["Nocc"]

    # calculate the eigenstates
    _, eigenvectors = bc.get_eigsystem(hfunc=hfunc, params=params, shift=shift)

    # calculate the overlap of the states
    links_x, _ = bc.get_links(eigenvectors=eigenvectors, Nocc=Nocc, direction="x")
    links_y, _ = bc.get_links(eigenvectors=eigenvectors, Nocc=Nocc, direction="y")

    # calculate wilson loops
    Wx = bc.get_wilsonloops(links=links_x, direction="x")
    Wy = bc.get_wilsonloops(links=links_y, direction="y")

    nu_x, _ = bc.get_Wilson_eigsystem(Wilsonloops=Wx)
    nu_y, _ = bc.get_Wilson_eigsystem(Wilsonloops=Wy)

    gapx0 = np.min(abs(nu_x[:, :, 1] - nu_x[:, :, 0]))
    gapx12 = np.min(abs(1 + nu_x[:, :, 0] - nu_x[:, :, 1]))
    gapy0 = np.min(abs(nu_y[:, :, 1] - nu_y[:, :, 0]))
    gapy12 = np.min(abs(1 + nu_y[:, :, 0] - nu_y[:, :, 1]))

    return [gapx0, gapx12, gapy0, gapy12]
