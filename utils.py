import numpy as np


def convolve_horizontal(image, hx):
    res = np.empty(np.shape(image))
    k = np.reshape(hx, np.shape(hx)[1])
    for i in range(np.shape(res)[0]):
        res[i] = np.convolve(image[i], k, mode='same')
    return res


def convolve_vertical(image, hy):
    res = np.empty(np.shape(image))
    k = np.reshape(hy, np.shape(hy)[0])
    for i in range(np.shape(res)[1]):
        res[:, i] = np.convolve(image[:, i], k, mode='same')
    return res


def hog(image):
    nwin_x = 6
    nwin_y = 6
    B = 9
    (L, C) = np.shape(image)
    H = np.zeros(shape=(nwin_x * nwin_y * B, 1))
    m = np.sqrt(L / 2.0)
    if C is 1:
        raise NotImplementedError
    step_x = np.floor(C / (nwin_x + 1))
    step_y = np.floor(L / (nwin_y + 1))
    cont = 0
    hx = np.array([[1, 0, -1]])
    hy = np.array([[-1], [0], [1]])
    grad_xr = convolve_horizontal(image, hx)
    grad_yu = convolve_vertical(image, hy)
    angles = np.arctan2(grad_yu, grad_xr)
    magnit = np.sqrt((grad_yu ** 2 + grad_xr ** 2))
    for n in range(nwin_y):
        for m in range(nwin_x):
            cont += 1
            angles2 = angles[int(n * step_y):int((n + 2) * step_y),
                      int(m * step_x):int((m + 2) * step_x)]
            magnit2 = magnit[int(n * step_y):int((n + 2) * step_y),
                      int(m * step_x):int((m + 2) * step_x)]
            v_angles = angles2.ravel()
            v_magnit = magnit2.ravel()
            bin = 0
            H2 = np.zeros(shape=(B, 1))

            for ang_lim in np.arange(start=-np.pi + 2 * np.pi / B,
                                     stop=np.pi + 2 * np.pi / B,
                                     step=2 * np.pi / B):
                check = v_angles < ang_lim
                v_angles = (v_angles * (~check)) + (check) * 100
                H2[bin] += np.sum(v_magnit * check)
                bin += 1

            H2 = H2 / (np.linalg.norm(H2) + 0.01)
            H[(cont - 1) * B:cont * B] = H2
    return H
