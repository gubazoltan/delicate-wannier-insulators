import numpy as np
import scipy as scp
import sympy as sp
from sympy.physics.quantum import TensorProduct

def get_layeredHamiltonian(ksymbols, symbols): 
    kx_sym, ky_sym = ksymbols
    alpha_sym, beta_sym, lambda_z, gamma_z = symbols

    #create the Pauli matrices
    s0 = sp.eye(2)
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])

    h_1d = (sp.sin(kx_sym) + alpha_sym * sp.sin(2*kx_sym)) * sx
    h_1d += (sp.cos(kx_sym) + beta_sym * sp.cos(2*kx_sym)) * sz

    H_layered = sp.Matrix(np.zeros((4,4)))
    H_layered += TensorProduct(sz,h_1d)
    H_layered += TensorProduct(sy,s0) * lambda_z * sp.sin(ky_sym)
    H_layered += TensorProduct(sx,s0) * (gamma_z + lambda_z * sp.cos(ky_sym))

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

def get_BZ(params, shift = False):
    # extract the parameters
    Nx, Ny = params["Nx"], params["Ny"]

    # define the brillouin zone
    Kxs = np.linspace(0, 2*np.pi, Nx, endpoint=False)
    Kys = np.linspace(0, 2*np.pi, Ny, endpoint=False)

    #shift the BZ to avoid high symmetry points
    if shift: 
        Kxs += 1e-10
        Kys += 1e-10

    # return the brillouin zone
    return Kxs, Kys

def get_eigsystem(hfunc, params, shift = False): 
    Nx, Ny, Nbands = params["Nx"], params["Ny"], params["Nbands"]

    Kxs, Kys = get_BZ(params, shift)

    #define container for the eigenvalues and eigenvectors
    eigenvalues = np.zeros((Nx, Ny, Nbands))
    eigenvectors = np.zeros((Nx, Ny, Nbands, Nbands), dtype = complex)

    #loop over the kx and ky values
    for i, kx in enumerate(Kxs):
        for j, ky in enumerate(Kys):
            #diagonalize the Hamiltonian
            evals, evecs = np.linalg.eigh(hfunc(kx, ky))

            #sort the eigenvalues and eigenvectors
            idx = evals.argsort()
            evals = evals[idx]
            evecs = evecs[:, idx]

            #store the eigenvalues and eigenvectors
            eigenvalues[i, j] = evals
            eigenvectors[i, j] = evecs

    #return the eigenvalues and eigenvectors
    return eigenvalues, eigenvectors

def get_links(eigenvectors, Nocc, direction = "x"): 
    #get the size of the system
    (Nx, Ny, Nbands, _) = eigenvectors.shape

    #define container for the links
    links_occ = np.zeros((Nx, Ny, Nocc, Nocc)).astype(np.complex128)
    links_unocc = np.zeros((Nx, Ny, Nbands-Nocc, Nbands-Nocc)).astype(np.complex128)

    if direction == "x":
        # loop over the brillouin zone
        for i in range(Nx):
            for j in range(Ny):
                # get the overlap matrix
                overlap_occ = (
                    eigenvectors[(i + 1) % Nx, j, :, :Nocc].conj().T
                    @ eigenvectors[i, j, :, :Nocc]
                )
                overlap_unocc = (
                    eigenvectors[(i + 1) % Nx, j, :, Nocc:].conj().T
                    @ eigenvectors[i, j, :, Nocc:]
                )

                # do the singular value decomposition
                U_occ, _, Vh_occ = np.linalg.svd(overlap_occ)
                U_unocc, _, Vh_unocc = np.linalg.svd(overlap_unocc)

                # store the links
                links_occ[i, j, :, :] = U_occ @ Vh_occ
                links_unocc[i, j, :, :] = U_unocc @ Vh_unocc

    elif direction == "y":
        # loop over the brillouin zone
        for i in range(Nx): 
            for j in range(Ny): 
                #get the overlap matrix
                overlap_occ = (
                    eigenvectors[i, (j + 1) % Ny, :, :Nocc].conj().T
                    @ eigenvectors[i, j, :, :Nocc]
                )
                overlap_unocc = (
                    eigenvectors[i, (j + 1) % Ny, :, Nocc:].conj().T
                    @ eigenvectors[i, j, :, Nocc:]
                )

                #do the singular value decomposition
                U_occ, _, Vh_occ = np.linalg.svd(overlap_occ)
                U_unocc, _, Vh_unocc = np.linalg.svd(overlap_unocc)

                #store the links
                links_occ[i, j, :, :] = U_occ @ Vh_occ
                links_unocc[i, j, :, :] = U_unocc @ Vh_unocc

    else:
        raise ValueError("Direction must be 'x' or 'y'")
    
    #return the links
    return links_occ, links_unocc

def get_wilsonloops(links, direction="x"):
    # get the parameters of the system
    (Nx, Ny, Noccunocc, _) = links.shape

    # define container for the wilson loops
    Wilsonloops = np.zeros((Nx, Ny, Noccunocc, Noccunocc)).astype(np.complex128)

    if direction == "x":
        # loop over the brillouin zone
        for i in range(Nx):
            for j in range(Ny):
                # define container for the Wilson loop
                W = np.eye(Noccunocc).astype(np.complex128)

                # iterate over the x direction
                for kp in range(Nx):
                    W = links[(i + kp) % Nx, j, :, :] @ W

                # store the Wilson loop
                Wilsonloops[i, j, :, :] = W

    elif direction == "y":
        # loop over the brillouin zone
        for i in range(Nx):
            for j in range(Ny):
                # define container for the Wilson loop
                W = np.eye(Noccunocc).astype(np.complex128)

                # iterate over the y direction
                for kp in range(Ny):
                    W = links[i, (j + kp) % Ny, :, :] @ W

                # store the Wilson loop
                Wilsonloops[i, j, :, :] = W

    else: 
        raise ValueError("Direction must be 'x' or 'y'")
    
    #return the Wilson loops
    return Wilsonloops

def get_Wilson_eigsystem(Wilsonloops):
    # get the parameters of the system
    (Nx, Ny, Noccunocc, _) = Wilsonloops.shape

    # define container for the eigenvalues and eigenvectors
    nu_vals = np.zeros((Nx, Ny, Noccunocc))
    nu_vecs = np.zeros((Nx, Ny, Noccunocc, Noccunocc)).astype(np.complex128)

    # loop over the brillouin zone
    for i in range(Nx):
        for j in range(Ny):
            # calculate the Wilson loop eigenvalues and eigenvectors
            evals, evecs = np.linalg.eig(Wilsonloops[i, j, :, :])
            angles = np.angle(evals) / (2 * np.pi)

            # sort the eigenvalues and eigenvectors
            idx = np.argsort(angles)
            angles = angles[idx]
            evecs = evecs[:, idx]

            # store the eigenvalues and eigenvectors
            nu_vals[i, j, :] = angles[:]
            nu_vecs[i, j, :, :] = evecs[:, :]

    # return the eigenvalues and eigenvectors
    return nu_vals, nu_vecs

def get_Wilson_eigsystem_schur(Wilsonloops):
    # get the parameters of the system
    (Nx, Ny, Noccunocc, _) = Wilsonloops.shape

    # define container for the eigenvalues and eigenvectors
    nu_vals = np.zeros((Nx, Ny, Noccunocc))
    nu_vecs = np.zeros((Nx, Ny, Noccunocc, Noccunocc)).astype(np.complex128)

    # loop over the brillouin zone
    for i in range(Nx):
        for j in range(Ny):
            # calculate the Wilson loop eigenvalues and eigenvectors
            T, Z = scp.linalg.schur(Wilsonloops[i, j, :, :])
            evals = scp.linalg.eigvals(T)
            angles = np.angle(evals) / (2 * np.pi)

            # sort the eigenvalues and eigenvectors
            idx = np.argsort(angles)
            angles = angles[idx]
            evecs = Z[:, idx]

            # store the eigenvalues and eigenvectors
            nu_vals[i, j, :] = angles[:]
            nu_vecs[i, j, :, :] = evecs[:, :]

    # return the eigenvalues and eigenvectors
    return nu_vals, nu_vecs

def slab_geometry(h_symbolic, ksymbols, params, direction = "x"):
    # extract the parameters
    Nx, Ny, Nbands = params["Nx"], params["Ny"], params["Nbands"]
    kx_sym, ky_sym = ksymbols

    if direction == "x": 
        #define the nearest and next-nearest neighbor hopping terms
        Lx_nn_pos = sp.integrate(h_symbolic * sp.exp(sp.I * kx_sym * (-1.0)), (kx_sym, -sp.pi, sp.pi))/ (2 * sp.pi)
        Lx_nn_neg = sp.integrate(h_symbolic * sp.exp(sp.I * kx_sym * 1.0), (kx_sym, -sp.pi, sp.pi))/ (2 * sp.pi)
        Lx_nnn_pos = sp.integrate(h_symbolic * sp.exp(sp.I * 2 * kx_sym * (-1.0)), (kx_sym, -sp.pi, sp.pi))/ (2 * sp.pi)
        Lx_nnn_neg = sp.integrate(h_symbolic * sp.exp(sp.I * 2 * kx_sym * 1.0), (kx_sym, -sp.pi, sp.pi))/ (2 * sp.pi)

        Lx_nn_pos = Lx_nn_pos.rewrite(sp.cos).simplify()
        Lx_nn_neg = Lx_nn_neg.rewrite(sp.cos).simplify()
        Lx_nnn_pos = Lx_nnn_pos.rewrite(sp.cos).simplify()
        Lx_nnn_neg = Lx_nnn_neg.rewrite(sp.cos).simplify()

        H_diag = h_symbolic - (Lx_nn_pos * sp.exp(sp.I * kx_sym) + Lx_nn_neg * sp.exp(-sp.I * kx_sym) + Lx_nnn_pos * sp.exp(sp.I * 2 * kx_sym) + Lx_nnn_neg * sp.exp(-sp.I * 2 * kx_sym))
        H_diag = H_diag.rewrite(sp.cos).simplify()
        H_diag = sp.nsimplify(H_diag, tolerance = 1e-8)

        #create container for the Hamiltonian
        h = sp.zeros(Nx*Nbands, Nx*Nbands)

        for i in range(Nx):
            h[i*Nbands:(i+1)*Nbands, i*Nbands:(i+1)*Nbands] = H_diag #+ 1e-3 * sy 

            #if downwards hopping is possible
            if i > 0: 
                h[(i-1)*Nbands:i*Nbands, i*Nbands:(i+1)*Nbands] = Lx_nn_pos[:,:]

            if i > 1:
                h[(i-2)*Nbands:(i-1)*Nbands, i*Nbands:(i+1)*Nbands] = Lx_nnn_pos[:,:]

            #if upwards hopping is possible
            if i < Nx - 1:
                h[(i+1)*Nbands:(i+2)*Nbands, i*Nbands:(i+1)*Nbands] = Lx_nn_neg[:,:]

            if i < Nx - 2:
                h[(i+2)*Nbands:(i+3)*Nbands, i*Nbands:(i+1)*Nbands] = Lx_nnn_neg[:,:]
        
        slab_h = sp.lambdify(ky_sym, h, "numpy")

    elif direction == "y":
        Ly_nn_pos = sp.integrate(h_symbolic * sp.exp(sp.I * ky_sym * (-1.0)), (ky_sym, -sp.pi, sp.pi))/ (2 * sp.pi)
        Ly_nn_neg = sp.integrate(h_symbolic * sp.exp(sp.I * ky_sym * 1.0), (ky_sym, -sp.pi, sp.pi))/ (2 * sp.pi)
        Ly_nnn_pos = sp.integrate(h_symbolic * sp.exp(sp.I * 2 * ky_sym * (-1.0)), (ky_sym, -sp.pi, sp.pi))/ (2 * sp.pi)
        Ly_nnn_neg = sp.integrate(h_symbolic * sp.exp(sp.I * 2 * ky_sym * 1.0), (ky_sym, -sp.pi, sp.pi))/ (2 * sp.pi)

        Ly_nn_pos = Ly_nn_pos.rewrite(sp.cos).simplify()
        Ly_nn_neg = Ly_nn_neg.rewrite(sp.cos).simplify()
        Ly_nnn_pos = Ly_nnn_pos.rewrite(sp.cos).simplify()
        Ly_nnn_neg = Ly_nnn_neg.rewrite(sp.cos).simplify()

        H_diag = h_symbolic - (Ly_nn_pos * sp.exp(sp.I * ky_sym) + Ly_nn_neg * sp.exp(-sp.I * ky_sym))
        H_diag -= (Ly_nnn_pos * sp.exp(sp.I * 2 * ky_sym) + Ly_nnn_neg * sp.exp(-sp.I * 2 * ky_sym))
        H_diag = H_diag.rewrite(sp.cos).simplify()
        H_diag = sp.nsimplify(H_diag, tolerance = 1e-8)
        
        #create container for the Hamiltonian
        h = sp.zeros(Ny*Nbands, Ny*Nbands)

        for i in range(Ny):
            h[i*Nbands:(i+1)*Nbands, i*Nbands:(i+1)*Nbands] = H_diag

            #if downwards hopping is possible
            if i > 0: 
                h[(i-1)*Nbands:i*Nbands, i*Nbands:(i+1)*Nbands] = Ly_nn_pos[:,:]

            if i > 1:
                h[(i-2)*Nbands:(i-1)*Nbands, i*Nbands:(i+1)*Nbands] = Ly_nnn_pos[:,:]

            #if upwards hopping is possible
            if i < Ny - 1:
                h[(i+1)*Nbands:(i+2)*Nbands, i*Nbands:(i+1)*Nbands] = Ly_nn_neg[:,:]

            if i < Ny - 2:
                h[(i+2)*Nbands:(i+3)*Nbands, i*Nbands:(i+1)*Nbands] = Ly_nnn_neg[:,:]
            
        slab_h = sp.lambdify(kx_sym, h, "numpy")

    return slab_h