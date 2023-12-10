import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

import warnings
warnings.filterwarnings("ignore")

from google.colab import drive
drive.mount('/content/drive')

path = "/content/drive/MyDrive/Image_Compression_PCA/Source_Images"

img_name = os.listdir(path)

source_img = cv.imread(path + "/" + img_name[0])
hx, hy, c = source_img.shape

resized_src = cv.resize(source_img, (10*int(hy/10), 10*int(hx/10)))

cv2_imshow(resized_src)

y_tiles = int(resized_src.shape[0]/10)
x_tiles = int(resized_src.shape[1]/10)

cb = np.empty((100, y_tiles*x_tiles))
cg = np.empty((100, y_tiles*x_tiles))
cr = np.empty((100, y_tiles*x_tiles))

index = 0

for i in range(y_tiles):
    for j in range(x_tiles):
        cb[:, index] = resized_src[10*i:10*(i + 1), 10*j:10*(j + 1), 0].flatten()
        cg[:, index] = resized_src[10*i:10*(i + 1), 10*j:10*(j + 1), 1].flatten()
        cr[:, index] = resized_src[10*i:10*(i + 1), 10*j:10*(j + 1), 2].flatten()
        index += 1

cb_mean = np.mean(cb, axis=1)
cg_mean = np.mean(cg, axis=1)
cr_mean = np.mean(cr, axis=1)

for i in range(cb.shape[1]):
    cb[:, i] = cb[:, i] - cb_mean
    cg[:, i] = cg[:, i] - cg_mean
    cr[:, i] = cr[:, i] - cr_mean

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

compress_to = 0.2
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

new_cb = np.empty(cb.shape)
new_cg = np.empty(cg.shape)
new_cr = np.empty(cr.shape)

def compute_array(weights, eigs):
    arr = np.zeros(100)
    for i in range(len(weights)):
        arr = arr + weights[i]*eigs[:,i]

    return arr

for i in range(new_cb.shape[1]):
    new_cb[:, i] = compute_array(weights_cb[:, i], eig_cb)
    new_cg[:, i] = compute_array(weights_cg[:, i], eig_cg)
    new_cr[:, i] = compute_array(weights_cr[:, i], eig_cr)

for i in range(cb.shape[1]):
    new_cb[:, i] = new_cb[:, i] + cb_mean
    new_cg[:, i] = new_cg[:, i] + cg_mean
    new_cr[:, i] = new_cr[:, i] + cr_mean

compr_img = np.empty(resized_src.shape, dtype="int")

index = 0
for i in range(y_tiles):
    for j in range(x_tiles):
        compr_img[i*10:(i + 1)*10, j*10:(j + 1)*10, 0] = np.reshape(new_cb[:,index], (10, 10))
        compr_img[i*10:(i + 1)*10, j*10:(j + 1)*10, 1] = np.reshape(new_cg[:,index], (10, 10))
        compr_img[i*10:(i + 1)*10, j*10:(j + 1)*10, 2] = np.reshape(new_cr[:,index], (10, 10))
        index += 1

cv2_imshow(compr_img)

cv.imwrite("/content/drive/MyDrive/Image_Compression_PCA/Output_Images/Prajakta_Mali7_" + str(new_dim) + ".jpg", compr_img)