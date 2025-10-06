import numpy as np
from scipy.special import comb

# otro stuff ---------------------
def BeBOT(N, T):

    tnodes = np.linspace(0, T, N + 1)    
    w = np.full(N + 1, T / (N + 1))
    Dm = BernsteinDifferentiationMatrix(N, T) @ DegElevMatrix(N - 1, N)
    
    return tnodes, w, Dm

def BernsteinDifferentiationMatrix(N, T):
    
    Dm = np.zeros((N + 1, N))
    np.fill_diagonal(Dm[:-1], -N / T)
    np.fill_diagonal(Dm[1:], N / T)
    
    return Dm

def BernsteinProduct(A, B):
    M = len(A) - 1
    N = len(B) - 1
    C = np.zeros(M + N + 1)

    # Loop through all k from 0 to M+N
    for k in range(M + N + 1):
        C_k = 0  # Initialize the sum for C[k]
        for j in range(max(0, k - N), min(M, k) + 1):
            C_k += comb(M, j, exact=True) * comb(N, k - j, exact=True) / comb(M + N, k, exact=True) * A[j] * B[k - j]
        C[k] = C_k

    return C

def DegElevMatrix(N, M):

    if M <= N:
        raise ValueError("M must be greater than N.")
    
    r = M - N  # Degree difference
    
    # Indices for i and j
    i_vals = np.arange(N + 1)  # Indices for i
    j_vals = np.arange(r + 1)  # Indices for j
    
    # Compute the combinations for i and j
    comb_N_i = comb(N, i_vals)  # Shape (N+1,)
    comb_r_j = comb(r, j_vals)  # Shape (r+1,)
    comb_M_ij = comb(M, i_vals[:, None] + j_vals)  # Shape (N+1, r+1)
    
    # Initialize the matrix with zeros
    E = np.zeros((M + 1, N + 1))
    
    # Fill the matrix following the desired pattern
    for i in range(N + 1):
        for j in range(r + 1):
            E[i + j, i] = comb_N_i[i] * comb_r_j[j] / comb_M_ij[i, j]
    
    return E.T
# --------------------------------

# compute polys
def BernsteinMatrix_a2b(N, t):
    
    B = np.ones(N + 1)
    if t.ndim == 1 or (t.shape[0] == 1 and t.shape[1] > 1):  # Check if it's a row vector
        t = t.reshape(-1, 1)  # Convert to column vector (1000, 1)

    t0 = t[0]
    tf = t[-1]
    
    for j in range(0,(N // 2)):
        B[j+1] = B[j] * (N - j) / (j+1)
        B[N - (j+1)] = B[j+1]
    for j in range(1, (N // 2) + 1):  # Python range is 0-based and exclusive of the upper bound
        B[j] = B[j - 1] * (N + 1 - j) / j
        B[N - j] = B[j]
    T = np.ones([t.size, N + 1])  # T matrix, size based on t
    TT = np.ones([t.size, N + 1])  # TT matrix, size based on t
    tp = t.reshape([t.size, 1]) - t0  # Adjust t by t0
    ttp = tf - t  # Adjust t by tf
    for j in range(1, N + 1):  # Loop from 1 to N (inclusive)
        T[:, j] = tp.flatten() * T[:, j - 1]  # Update T[:, j] based on tp and the previous column
        T[:, j - 1] = T[:, j - 1] * B[j - 1]  # Multiply the previous column by the Bernstein coefficient
        TT[:, N - j] = ttp.flatten() * TT[:, N - j + 1]  # Update TT from the back
    
    b = T * TT / ((tf - t0) ** N)
    return b

# single bernstein, time -
def BernsteinPoly(Cp, t):

    t0 = t[0]
    tf = t[-1]
    Cp = Cp.reshape(1,-1)
    M, N = Cp.shape

    if N == 1 and M > 1:
        Cp = Cp.T
        N = M

    N = N - 1

    B = BernsteinMatrix_a2b(N, t).T
    poly_t = np.dot(Cp, B)

    return poly_t.T, B

# output: evaluated polynomial
def PiecewiseBernsteinPoly(Cp, tknots, t):
    M = len(tknots) - 1
    dim, totN = Cp.shape

    if totN == 1:
        Cp = Cp.T
    
    _, totN = Cp.shape
    N = totN // M - 1  

    poly_t = np.zeros((dim, len(t)))

    for i in range(M):
        for k, t in enumerate(t):
            if tknots[i] <= t <= tknots[i + 1]:
                t_range = np.array([tknots[i], tknots[i + 1]])
                if i < M - 1:
                    poly_t[:, k], _ = BernsteinPoly(Cp[:, (i - 1) * N + i : i * N + i], t_range)
                else:
                    poly_t[:, k], _ = BernsteinPoly(Cp[:, (i - 1) * N + i :], t_range)
                    
    return poly_t

# blob shi ----------------------
def BTDSolid_DegEl(Cp, k, n): # not necessarily optimized for time, though performs well
    nk = n[k]
    if Cp.ndim == 5:
        if k == 0:
            CpE = np.zeros((Cp.shape[0], n[0] + 2, n[1] + 1, n[2] + 1, n[3] + 1))
            CpE[:, 0, :, :, :] = Cp[:, 0, :, :, :]
            CpE[:, nk+1, :, :, :] = Cp[:, -1, :, :, :]
        elif k == 1:
            CpE = np.zeros((Cp.shape[0], n[0] + 1, n[1] + 2, n[2] + 1, n[3] + 1))
            CpE[:, :, 0, :, :] = Cp[:, :, 0, :, :]
            CpE[:, :, nk+1, :, :] = Cp[:, :, -1, :, :]
        elif k == 2:
            CpE = np.zeros((Cp.shape[0], n[0] + 1, n[1] + 1, n[2] + 2, n[3] + 1))
            CpE[:, :, :, 0, :] = Cp[:, :, :, 0, :]
            CpE[:, :, :, nk+1, :] = Cp[:, :, :, -1, :]
        else:
            CpE = np.zeros((Cp.shape[0], n[0] + 1, n[1] + 1, n[2] + 1, n[3] + 2))
            CpE[:, :, :, :, 0] = Cp[:, :, :, :, 0]
            CpE[:, :, :, :, nk+1] = Cp[:, :, :, :, -1]
        
        for ik in range(1, nk + 1):
            if k == 0:
                CpE[:, ik, :, :, :] = (ik / (nk + 1)) * Cp[:, ik - 1, :, :, :] + (1 - (ik / (nk + 1))) * Cp[:, ik, :, :, :]
            elif k == 1:
                CpE[:, :, ik, :, :] = (ik / (nk + 1)) * Cp[:, :, ik - 1, :, :] + (1 - (ik / (nk + 1))) * Cp[:, :, ik, :, :]
            elif k == 2:
                CpE[:, :, :, ik, :] = (ik / (nk + 1)) * Cp[:, :, :, ik - 1, :] + (1 - (ik / (nk + 1))) * Cp[:, :, :, ik, :]
            else:
                CpE[:, :, :, :, ik] = (ik / (nk + 1)) * Cp[:, :, :, :, ik - 1] + (1 - (ik / (nk + 1))) * Cp[:, :, :, :, ik]
                
    if Cp.ndim == 4:
        if k == 0:
            CpE = np.zeros((n[0] + 2, n[1] + 1, n[2] + 1, n[3] + 1))
            CpE[0, :, :, :] = Cp[0, :, :, :]
            CpE[nk+1, :, :, :] = Cp[-1, :, :, :]
        elif k == 1:
            CpE = np.zeros((n[0] + 1, n[1] + 2, n[2] + 1, n[3] + 1))
            CpE[:, 0, :, :] = Cp[:, 0, :, :]
            CpE[:, nk+1, :, :] = Cp[:, -1, :, :]
        elif k == 2:
            CpE = np.zeros((n[0] + 1, n[1] + 1, n[2] + 2, n[3] + 1))
            CpE[:, :, 0, :] = Cp[:, :, 0, :]
            CpE[:, :, nk+1, :] = Cp[:, :, -1, :]
        else:
            CpE = np.zeros((n[0] + 1, n[1] + 1, n[2] + 1, n[3] + 2))
            CpE[:, :, :, 0] = Cp[:, :, :, 0]
            CpE[:, :, :, nk+1] = Cp[:, :, :, -1]
        
        for ik in range(1, nk + 1):
            if k == 0:
                CpE[ik, :, :, :] = (ik / (nk + 1)) * Cp[ik - 1, :, :, :] + (1 - (ik / (nk + 1))) * Cp[ik, :, :, :]
            elif k == 1:
                CpE[:, ik, :, :] = (ik / (nk + 1)) * Cp[:, ik - 1, :, :] + (1 - (ik / (nk + 1))) * Cp[:, ik, :, :]
            elif k == 2:
                CpE[:, :, ik, :] = (ik / (nk + 1)) * Cp[:, :, ik - 1, :] + (1 - (ik / (nk + 1))) * Cp[:, :, ik, :]
            else:
                CpE[:, :, :, ik] = (ik / (nk + 1)) * Cp[:, :, :, ik - 1] + (1 - (ik / (nk + 1))) * Cp[:, :, :, ik]
                
    return CpE

def BTDSolid_mult(Cp1, Cp2, n, m):
    n0, n1, n2, n3 = n
    m0, m1, m2, m3 = m

    # Initialize U with zeros
    U = np.zeros((n0 + m0 + 1, n1 + m1 + 1, n2 + m2 + 1, n3 + m3 + 1))
    
    # Precompute combination arrays for exact=True manually
    comb_n0 = [comb(n0, j, exact=True) for j in range(n0 + 1)]
    comb_n1 = [comb(n1, j, exact=True) for j in range(n1 + 1)]
    comb_n2 = [comb(n2, j, exact=True) for j in range(n2 + 1)]
    comb_n3 = [comb(n3, j, exact=True) for j in range(n3 + 1)]
    
    comb_m0 = [comb(m0, j, exact=True) for j in range(m0 + 1)]
    comb_m1 = [comb(m1, j, exact=True) for j in range(m1 + 1)]
    comb_m2 = [comb(m2, j, exact=True) for j in range(m2 + 1)]
    comb_m3 = [comb(m3, j, exact=True) for j in range(m3 + 1)]
    
    comb_nm0 = [comb(n0 + m0, j, exact=True) for j in range(n0 + m0 + 1)]
    comb_nm1 = [comb(n1 + m1, j, exact=True) for j in range(n1 + m1 + 1)]
    comb_nm2 = [comb(n2 + m2, j, exact=True) for j in range(n2 + m2 + 1)]
    comb_nm3 = [comb(n3 + m3, j, exact=True) for j in range(n3 + m3 + 1)]
    
    # Main computation
    for i0 in range(n0 + m0 + 1):
        for i1 in range(n1 + m1 + 1):
            for i2 in range(n2 + m2 + 1):
                for i3 in range(n3 + m3 + 1):
                    for j0 in range(max(0, i0 - m0), min(n0, i0) + 1):
                        for j1 in range(max(0, i1 - m1), min(n1, i1) + 1):
                            for j2 in range(max(0, i2 - m2), min(n2, i2) + 1):
                                for j3 in range(max(0, i3 - m3), min(n3, i3) + 1):
                                    # Calculate numerator and denominator
                                    num = (comb_n0[j0] * comb_n1[j1] * comb_n2[j2] * comb_n3[j3] *
                                           comb_m0[i0 - j0] * comb_m1[i1 - j1] * comb_m2[i2 - j2] * comb_m3[i3 - j3])
                                    den = (comb_nm0[i0] * comb_nm1[i1] * comb_nm2[i2] * comb_nm3[i3])
                                    # Update U
                                    U[i0, i1, i2, i3] += (Cp1[j0, j1, j2, j3] *
                                                          Cp2[i0 - j0, i1 - j1, i2 - j2, i3 - j3] * num / den)
    return U

def BTDSolid_partial(Cp, k, Xkf, n):
    nk = n[k]
    if Cp.ndim == 4:
        # Initialize the dCp array based on the condition for k
        shape = list(n)
        if k == 0:
            shape[1], shape[2], shape[3] = n[1] + 1, n[2] + 1, n[3] + 1
        elif k == 1:
            shape[0], shape[2], shape[3] = n[0] + 1, n[2] + 1, n[3] + 1
        elif k == 2:
            shape[0], shape[1], shape[3] = n[0] + 1, n[1] + 1, n[3] + 1
        else:
            shape[0], shape[1], shape[2] = n[0] + 1, n[1] + 1, n[2] + 1
        
        dCp = np.zeros(shape)
        
        # Calculate the differences in Cp slices using broadcasting
        if k == 0:
            dCp[:nk, :, :, :] = (nk / Xkf) * (Cp[1:nk+1, :, :, :] - Cp[:nk, :, :, :])
        elif k == 1:
            dCp[:, :nk, :, :] = (nk / Xkf) * (Cp[:, 1:nk+1, :, :] - Cp[:, :nk, :, :])
        elif k == 2:
            dCp[:, :, :nk, :] = (nk / Xkf) * (Cp[:, :, 1:nk+1, :] - Cp[:, :, :nk, :])
        else:
            dCp[:, :, :, :nk] = (nk / Xkf) * (Cp[:, :, :, 1:nk+1] - Cp[:, :, :, :nk])
            
    if Cp.ndim == 5:
        # Initialize the dCp array based on the condition for k
        shape = list([3,*n])
        if k == 0:
            shape[2], shape[3], shape[4] = n[1] + 1, n[2] + 1, n[3] + 1
        elif k == 1:
            shape[1], shape[3], shape[4] = n[0] + 1, n[2] + 1, n[3] + 1
        elif k == 2:
            shape[1], shape[2], shape[4] = n[0] + 1, n[1] + 1, n[3] + 1
        else:
            shape[1], shape[2], shape[3] = n[0] + 1, n[1] + 1, n[2] + 1
        
        dCp = np.zeros(shape)
        
        # Calculate the differences in Cp slices using broadcasting
        if k == 0:
            dCp[:, :nk, :, :, :] = (nk / Xkf) * (Cp[:, 1:nk+1, :, :, :] - Cp[:, :nk, :, :, :])
        elif k == 1:
            dCp[:, :, :nk, :, :] = (nk / Xkf) * (Cp[:, :, 1:nk+1, :, :] - Cp[:, :, :nk, :, :])
        elif k == 2:
            dCp[:, :, :, :nk, :] = (nk / Xkf) * (Cp[:, :, :, 1:nk+1, :] - Cp[:, :, :, :nk, :])
        else:
            dCp[:, :, :, :, :nk] = (nk / Xkf) * (Cp[:, :, :, :, 1:nk+1] - Cp[:, :, :, :, :nk])
    return dCp

def BTDSolid_CGDT(r, n):
    n0, n1, n2, n3 = n
    
    # Initialize F with zeros
    F = np.zeros((3, 3, n0 + 1, n1 + 1, n2 + 1, n3 + 1))
    
    # Loop through i and k to compute F
    for i in range(3):
        for k in range(3):
            # Compute partial derivative
            drdX = BTDSolid_partial(r[i, :, :, :, :], k, 1, n)
            
            # Adjust dimensions for DegEl
            nprime = n.copy()
            nprime[k] -= 1
            
            # Compute Fik and assign it to F
            F[i, k, :, :, :, :] = BTDSolid_DegEl(drdX, k, nprime)
    
    # Initialize C
    C = np.zeros((2 * n0 + 1, 2 * n1 + 1, 2 * n2 + 1, 2 * n3 + 1))
    
    # Loop through k to compute C
    for k in range(3):
        Fk = F[:, k, :, :, :, :]
        
        # Reshape Fk only once outside the loop
        Fk1, Fk2, Fk3 = [np.reshape(Fk[i], (n0 + 1, n1 + 1, n2 + 1, n3 + 1)) for i in range(3)]
        
        # Compute C in one step using vectorized operations
        C += (BTDSolid_mult(Fk1, Fk1, n, n) + 
              BTDSolid_mult(Fk2, Fk2, n, n) + 
              BTDSolid_mult(Fk3, Fk3, n, n))
    
    return C
# --------------------------------
