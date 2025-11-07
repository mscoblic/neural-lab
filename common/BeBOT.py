import numpy as np


def PiecewiseBernsteinPoly(Cp, tknots, time):
    """
    Evaluate piecewise Bernstein polynomial

    Cp: control points (can be 1D or 2D array)
    tknots: knot sequence
    time: evaluation time points
    """
    M = len(tknots) - 1

    # Handle 1D vs 2D input
    if Cp.ndim == 1:
        Cp = Cp.reshape(1, -1)

    dim, totN = Cp.shape
    N = int(totN / M - 1)  # Degree of each segment

    poly_t = np.zeros((dim, len(time)))

    for i in range(M):
        for k in range(len(time)):
            t = time[k]
            if tknots[i] <= t <= tknots[i + 1]:
                if i < M - 1:
                    # Slice for segment i
                    start_idx = i * N + i
                    end_idx = (i + 1) * N + i + 1
                    poly_t[:, k] = BernsteinPoly(Cp[:, start_idx:end_idx], t, tknots[i], tknots[i + 1])
                else:
                    # Last segment
                    start_idx = i * N + i
                    poly_t[:, k] = BernsteinPoly(Cp[:, start_idx:], t, tknots[i], tknots[i + 1])

    return poly_t


def BernsteinPoly(Cp, t, t0=0, tf=1):
    """
    Evaluate Bernstein polynomial

    Cp: control points matrix (n x N+1)
    t: time (scalar or array)
    t0, tf: time interval
    """
    M, N = Cp.shape

    # Transpose if needed
    if N == 1 and M > 1:
        Cp = Cp.T
        N = M

    N = N - 1  # Degree

    # Get Bernstein basis matrix
    B = BernsteinMatrix_a2b(N, t, t0, tf).T

    # Evaluate polynomial
    poly_t = Cp @ B

    return poly_t.flatten() if poly_t.shape[0] == 1 else poly_t


def BernsteinMatrix_a2b(N, t, tmin=0, tmax=1):
    """
    Compute Bernstein basis matrix

    N: polynomial degree
    t: time points (scalar or array)
    tmin, tmax: time interval
    """
    # Convert scalar to array if needed
    if np.isscalar(t):
        t = np.array([t])
    else:
        t = np.asarray(t)

    if len(t) > 1:
        tmax = t[-1]
        tmin = t[0]

    # Compute binomial coefficients B(j) = binomial(N, j)
    B = np.ones(N + 1)
    for j in range(1, int(np.ceil(N / 2)) + 1):
        B[j] = B[j - 1] * (N + 1 - j) / j
        B[N - j] = B[j]

    # Initialize T and TT matrices
    T = np.ones((len(t), N + 1))
    TT = np.ones((len(t), N + 1))

    tp = t - tmin
    ttp = tmax - t

    # Compute powers and apply binomial coefficients
    for j in range(N):
        T[:, j + 1] = tp * T[:, j]
        T[:, j] = T[:, j] * B[j]
        TT[:, N - 1 - j] = ttp * TT[:, N - j]  # FIXED: N-1-j instead of N-j

    # Final column uses last binomial coefficient
    T[:, N] = T[:, N] * B[N]

    # Compute final basis matrix
    b = T * TT / ((tmax - tmin) ** N)

    return b