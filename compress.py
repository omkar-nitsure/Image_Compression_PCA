import numpy as np
from reconstruct import reconstruct_img
from eigenvecs_proj import compute_eigs

# finally compresses the image as desired
def compress_img(cb, cg, cr, cb_mean, cg_mean, cr_mean, x_tiles, y_tiles, resized_src):

    compress_ratios = [0.01, 0.05, 0.2]

    compr_imgs = []

    for i in range(len(compress_ratios)):

        eig_cb, eig_cg, eig_cr, weights_cb, weights_cg, weights_cr = compute_eigs(cb, cg, cr, compress_ratios[i])

        new_cb = np.empty(cb.shape)
        new_cg = np.empty(cg.shape)
        new_cr = np.empty(cr.shape)

        for i in range(new_cb.shape[1]):
            new_cb[:, i] = reconstruct_img(weights_cb[:, i], eig_cb)
            new_cg[:, i] = reconstruct_img(weights_cg[:, i], eig_cg)
            new_cr[:, i] = reconstruct_img(weights_cr[:, i], eig_cr)

        for i in range(cb.shape[1]):
            new_cb[:, i] = new_cb[:, i] + cb_mean
            new_cg[:, i] = new_cg[:, i] + cg_mean
            new_cr[:, i] = new_cr[:, i] + cr_mean

        compr_img = np.empty(resized_src.shape, dtype="float")

        index = 0
        for i in range(y_tiles):
            for j in range(x_tiles):
                compr_img[i*10:(i + 1)*10, j*10:(j + 1)*10, 0] = np.reshape(new_cb[:,index], (10, 10))
                compr_img[i*10:(i + 1)*10, j*10:(j + 1)*10, 1] = np.reshape(new_cg[:,index], (10, 10))
                compr_img[i*10:(i + 1)*10, j*10:(j + 1)*10, 2] = np.reshape(new_cr[:,index], (10, 10))
                index += 1

        compr_imgs.append(compr_img)

    return compr_imgs