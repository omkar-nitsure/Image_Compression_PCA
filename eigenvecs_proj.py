import numpy as np


# computes the eigenvectors and the eigen-coefficients
def compute_eigs(cb, cg, cr, compress_to):

    cov_mat_cb = np.cov(cb)
    cov_mat_cg = np.cov(cg)
    cov_mat_cr = np.cov(cr)

    eig_vals_cb, eig_vecs_cb = np.linalg.eig(cov_mat_cb)
    eig_vals_cg, eig_vecs_cg = np.linalg.eig(cov_mat_cg)
    eig_vals_cr, eig_vecs_cr = np.linalg.eig(cov_mat_cr)

    eig_vals_cb = np.abs(eig_vals_cb)
    eig_vals_cg = np.abs(eig_vals_cg)
    eig_vals_cr = np.abs(eig_vals_cr)

    b = np.arange(0, 100, 1)
    g = np.arange(0, 100, 1)
    r = np.arange(0, 100, 1)

    b = np.array([b for _, b in sorted(zip(eig_vals_cb, b))])
    g = np.array([g for _, g in sorted(zip(eig_vals_cg, g))])
    r = np.array([r for _, r in sorted(zip(eig_vals_cr, r))])

    id_b = np.zeros(100, dtype="int")
    id_g = np.zeros(100, dtype="int")
    id_r = np.zeros(100, dtype="int")
    for i in range(100):
        id_b[i] = b[99 - i]
        id_g[i] = g[99 - i]
        id_r[i] = r[99 - i]

    new_dim = int(100*compress_to)

    eig_cb = np.empty((100, new_dim))
    eig_cg = np.empty((100, new_dim))
    eig_cr = np.empty((100, new_dim))

    for i in range(new_dim):
        eig_cb[:,i] = eig_vecs_cb[:,id_b[i]]
        eig_cg[:,i] = eig_vecs_cg[:,id_g[i]]
        eig_cr[:,i] = eig_vecs_cr[:,id_r[i]]

    weights_cb = np.empty((new_dim, cb.shape[1]))
    weights_cg = np.empty((new_dim, cg.shape[1]))
    weights_cr = np.empty((new_dim, cr.shape[1]))

    for i in range(cb.shape[1]):
        comps = []
        for j in range(eig_cb.shape[1]):
            comps.append(np.dot(cb[:,i], eig_cb[:,j]))
        comps = np.array(comps)
        weights_cb[:, i] = comps

    for i in range(cg.shape[1]):
        comps = []
        for j in range(eig_cg.shape[1]):
            comps.append(np.dot(cg[:,i], eig_cg[:,j]))
        comps = np.array(comps)
        weights_cg[:, i] = comps

    for i in range(cr.shape[1]):
        comps = []
        for j in range(eig_cr.shape[1]):
            comps.append(np.dot(cr[:,i], eig_cr[:,j]))
        comps = np.array(comps)
        weights_cr[:, i] = comps

    return eig_cb, eig_cg, eig_cr, weights_cb, weights_cg, weights_cr