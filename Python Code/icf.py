import lvm as lvm
import myGraph as myGraph
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


# Return an estimation of the noisy incomplete matrix m via iterative collaborative filtering.
# e is a list of pairs of indices corresponding to the observed entries.
# m is a matrix of observations where unobserved entries are equal to zero.
# kappa is the sparsity parameter.
def icf(e, m, kappa):
    if kappa <= 0 or kappa >= 1/2:
        raise ValueError("The sparsity parameter kappa must be strictly between 0 and 1/2.")

    [n, c] = m.shape
    if n != c:
        raise TypeError("The observed matrix m must be square.")

    # Define parameters.
    p = n ** (-1 + kappa)
    t = np.floor(np.log(1/p) / np.log(n*p))
    t = t.astype(int)
    rho = kappa/2
    eta = n ** (-1/2 * (kappa - rho))

    # Do sample splitting.
    e_1 = []
    e_2 = []
    e_3 = []
    for idx in range(len(e)):
        u = np.random.uniform()
        if u < 1/4:
            e_1.append(e[idx])
        elif u < 1/2:
            e_2.append(e[idx])
        else:
            e_3.append(e[idx])
    p_prime = p / (4 - p)
    e_1_ind = []
    for idx in range(len(e_1)):
        if np.random.uniform() < p_prime:
            e_1_ind.append(e_1[idx])

    # Do noisy nearest neighbors.
    g = myGraph.DiGraph()
    for idx in range(len(e_1)):
        (i, j) = e_1[idx]
        g.add_edge((i, j, m[i, j]))
    # The first index of n_tilde is the root node; the next two indices are the normalized nearest neighbor
    # computed from the nnb() method in myGraph
    n_tilde = np.zeros((n, 2, n))
    for u in range(n):
        n_tilde[u, :, :] = g.nnb(u, t, n)

    # Compute distances.
    m_2 = np.zeros((n, n))
    for idx in range(len(e_2)):
        m_2[e_2[idx]] = m[e_2[idx]]

    m_1_ind = np.zeros((n, n))
    for idx in range(len(e_1_ind)):
        m_1_ind[e_1_ind[idx]] = m[e_1_ind[idx]]

    d_hat = np.zeros((n, n))
    for u in range(n):
        for v in range(n):
            d_hat[u, v] = 1 / p_prime * (n_tilde[u, 0, :] - n_tilde[v, 0, :]) \
                          @ (m_2 + m_1_ind) @ (n_tilde[u, 1, :] - n_tilde[v, 1, :])
            # Take the absolute value. (Its absolute value can only be closer to the true d, which is nonnegative.)
            d_hat[u, v] = np.abs(d_hat[u, v])
    #print(d_hat)

    f_hat = np.zeros((n, n))
    for u in range(n):
        for v in range(n):
            total = 0
            count = 0
            for idx in range(len(e_3)):
                (a, b) = e_3[idx]
                if d_hat[u, a] < eta and d_hat[v, b] < eta:
                    total += m[a, b]
                    count += 1
            # Put a zero entry if e'''_uv is empty in Equation (3).
            if count > 0:
                f_hat[u, v] = total / count

    return f_hat


if __name__ == "__main__":
    # Unit test.
    n = 50
    r = 2
    f = lvm.lvm(n, r)

    kappa = 0.49
    p = n ** (-1 + kappa)

    # e is a list of pairs of indices corresponding to the observed entries.
    e = []
    for i in range(n):
        for j in range(n):
            if np.random.uniform() < p:
                e.append((i, j))

    # m is a noisy signal of f
    m = np.zeros((n, n))
    for idx in range(len(e)):
       (i, j) = e[idx]
       # Note: here the noise in creating m depends on f.
       dist_to_edge = min(f[i, j], 1 - f[i, j])
       m[i, j] = f[i, j] + np.random.uniform(-dist_to_edge, dist_to_edge)

    f_hat = icf(e, m, kappa)
    print(f)
    print(f_hat)
    print(f - f_hat)
