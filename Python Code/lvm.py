import numpy as np


# Return the value of the jth Haar wavelet at x.
# The 0th Haar wavelet is the constant function 1. The 1st Haar wavelet is the mother wavelet.
def haar(j, x):
    if x < 0 or x > 1:
        raise ValueError("The basis function is defined on [0,1].")

    if j == 0:
        return 1

    # Convert j to (n,k) indices according to Wikipedia's notation.
    n = np.floor(np.log2(j))
    k = j - 2 ** n

    val = None
    x_scaled = 2 ** n * x - k
    if x_scaled < 0:
        val = 0
    elif x_scaled < 0.5:
        val = 1
    elif x_scaled < 1:
        val = -1
    else:
        val = 0

    return 2 ** (n / 2) * val


# Return an n by n random matrix according to the latent variable model with n latent features and rank r.
def lvm(n, r):
    # Ignore the 0th wavelet for now.
    wavelet_idx = range(1, r)

    theta = np.random.uniform(0, 1, n)
    spectrum = np.diag(np.random.uniform(-1, 1, r - 1))

    q = np.empty((r - 1, n))
    for k in range(r - 1):
        for u in range(n):
            q[k, u] = haar(wavelet_idx[k], theta[u])
    f = q.T @ spectrum @ q

    # Shift and scale the entries of f so that they lie in [0,1].
    # This corresponds to adding the 0th wavelet (constant wavelet) and scaling the spectrum.
    f = (f - f.min()) / (f.max() - f.min())

    return f


if __name__ == "__main__":
    # Unit test.
    n = 10
    r = 8
    print(lvm(n, r))
