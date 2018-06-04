# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import pickle as pkl
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


def predict(x):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 35} i oznacza znak rozpoznany na danym przykladzie.
    :param x: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """
    f = open('data.pkl', 'rb')
    w = pkl.load(f)
    res = hog_extraction(x)
    return prediction(res, w)


def feature_extraction(x):
    """
    Ekstrakcja cech polegajaca na uśrednieniu kwadratów rozmiaru 4x4 na obrazie
    :param x: pełne obrazy
    :return: ekstrakcja cech z obrazów
    """
    n = np.shape(x)[0]
    res = np.empty((n, 14, 14))
    tmp = 1 - np.reshape(x, (n, 56, 56))

    for i in range(14):
        for j in range(14):
            res[:, i, j] = np.sum(tmp[:, 4*i:4*i + 4, 4*j:4*j + 4], axis=(1, 2)) / 16

    return np.reshape(res, (n, 14*14))


def hog_extraction(x):
    n = np.shape(x)[0]
    tmp = np.reshape(x, (n, 56, 56))

    res = np.empty((n, 324))

    for i in range(n):
        # print(i)
        res[i] = hog(tmp[i]).reshape(324)

    return res


def softmax(x):
    """
    :param x: macierz wejsciowych wartosci, NxK
    :return: macierz wartosci funkcji softmax dla wejscia x, NxK
    """
    L = np.exp(x)
    return L / np.sum(L, axis=1, keepdims=True)


def prediction(x, w):
    """
    :param x: ciag danych wejsciowych, NxM
    :param w: parametry modelu, KxM
    :return: wektor klas przewidzianych dla tych danych, Nx1
    """
    P = softmax(x @ w.transpose())
    return np.reshape(np.argmax(P, axis=1), (np.shape(x)[0], 1))


def measure(y_true, y_pred):
    """
    :param y_true: wektor prawdziwych klas, Nx1
    :param y_pred: wektor przewidzianych klas, Nx1
    :return: odsetek poprawnych klasyfikacji
    """
    return np.sum(y_true == y_pred)


# print("x")
#
# train_file = open('train.pkl', 'rb')
# data = pkl.load(train_file)
# train_file.close()
#
# X = data[0]
# Y = data[1]
# x_test = X[24000:]
# y_test = Y[24000:]
# y_pred = predict(x_test)
# test = measure(y_test, y_pred)
#
# print(test / np.shape(y_test)[0])
