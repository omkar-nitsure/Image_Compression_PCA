import numpy as np

# recomputes the vector in a lossy way
def reconstruct_img(weights, eigs):

    arr = np.zeros(100)
    for i in range(len(weights)):
        arr = arr + weights[i]*eigs[:,i]

    return arr