import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from preprocess import preproc_img
from reconstruct import reconstruct_img
from compress import compress_img
from eigenvecs_proj import compute_eigs

img = plt.imread("Inputs/Example_1.jpg")

cb, cg, cr, cb_mean, cg_mean, cr_mean, x_tiles, y_tiles, src = preproc_img(img)

compressed_imgs = compress_img(cb, cg, cr, cb_mean, cg_mean, cr_mean, x_tiles, y_tiles, src)

compressed_imgs = np.array(compressed_imgs)/255.0

fig = plt.figure(figsize=(16, 8))

ax = fig.add_subplot(1, 3, 1)
plt.title("lossy image compression")
plt.imshow(compressed_imgs[0])

ax = fig.add_subplot(1, 3, 2)
plt.title("medium lossy compression")
plt.imshow(compressed_imgs[1])

ax = fig.add_subplot(1, 3, 3)
plt.title("loss-less compression")
plt.imshow(compressed_imgs[2])

plt.suptitle("Different types of compression")

plt.savefig("Outputs/Output_Ex1.jpg")