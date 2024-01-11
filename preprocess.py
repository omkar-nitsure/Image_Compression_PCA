import cv2 as cv
import numpy as np


# function to preprocess the input image
def preproc_img(img):

    hx, hy, c = img.shape

    src = cv.resize(img, (10*int(hy/10), 10*int(hx/10)))

    y_tiles = int(src.shape[0]/10)
    x_tiles = int(src.shape[1]/10)

    cb = np.empty((100, y_tiles*x_tiles), dtype="float")
    cg = np.empty((100, y_tiles*x_tiles), dtype="float")
    cr = np.empty((100, y_tiles*x_tiles), dtype="float")

    index = 0

    for i in range(y_tiles):
        for j in range(x_tiles):
            cb[:, index] = src[10*i:10*(i + 1), 10*j:10*(j + 1), 0].flatten()
            cg[:, index] = src[10*i:10*(i + 1), 10*j:10*(j + 1), 1].flatten()
            cr[:, index] = src[10*i:10*(i + 1), 10*j:10*(j + 1), 2].flatten()
            index += 1

    cb_mean = np.mean(cb, axis=1, dtype="float")
    cg_mean = np.mean(cg, axis=1, dtype="float")
    cr_mean = np.mean(cr, axis=1, dtype="float")

    for i in range(cb.shape[1]):
        cb[:, i] = cb[:, i] - cb_mean
        cg[:, i] = cg[:, i] - cg_mean
        cr[:, i] = cr[:, i] - cr_mean

    return cb, cg, cr, cb_mean, cg_mean, cr_mean, x_tiles, y_tiles, src