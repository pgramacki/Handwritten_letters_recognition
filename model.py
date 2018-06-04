# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import pickle as pkl
import numpy as np
from utils import hog
from random import shuffle


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
        print(i)
        res[i] = hog(tmp[i]).reshape(324)

    return res


def picture_negation(x):
    return 1 - x


def one_hot(y):
    """
    :param y: wektor klas o wymiarach Nx1
    :return: macierz klas zakodowanych one-hot Nx36
    """
    N = np.shape(y)[0]
    res = np.zeros((N, 36), dtype=int)
    for i in range(N):
        res[i, y[i]] = 1
    return res


def softmax(x):
    """
    :param x: macierz wejsciowych wartosci, NxK
    :return: macierz wartosci funkcji softmax dla wejscia x, NxK
    """
    L = np.exp(x)
    return L / np.sum(L, axis=1, keepdims=True)


def regularized_likelihood_function_gradient(w, x_train, y_train, reg_lambda):
    """
    :param w: parametry modelu, KxM
    :param x_train: ciag treningowy wejscia, NxM
    :param y_train: ciag treningowy wyjscia, NxK
    :param reg_lambda: parametr regularyzacji
    :return: gradient funkcji logistycznej w regularyzacja l2 po w, KxM
    """
    w0 = np.array(w)
    # w0[0] = 0
    return (y_train - softmax(x_train @ w.transpose())).transpose() @ x_train + reg_lambda * w0


def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    """
    :param obj_fun: funkcja celu, ktora ma byc optymalizowana, grad = obj_fun(w, x, y)
    :param x_train: ciag treningowy wejscia
    :param y_train: ciag treningowy wyjscia
    :param w0: poczatkowe parametry w
    :param epochs: ilosc iteracji algorytmu
    :param eta: krok uczenia
    :param mini_batch: rozmiar mini_batcha
    :return: znalezione optymalne parametry w
    """
    N = np.shape(y_train)[0]
    w = np.array(w0)
    for e in range(epochs):
        # print(e)
        # order = np.arange(N)
        # shuffle(order)
        # x = x_train[order]
        # y = y_train[order]
        for m in range(0, N, mini_batch):
            grad = obj_fun(w, x_train[m:m + mini_batch, :], y_train[m:m + mini_batch, :])
            w += eta * grad

    return w


def measure(y_true, y_pred):
    """
    :param y_true: wektor prawdziwych klas, Nx1
    :param y_pred: wektor przewidzianych klas, Nx1
    :return: odsetek poprawnych klasyfikacji
    """
    x = y_true == y_pred
    s = np.sum(x)
    return np.sum(y_true == y_pred)


def prediction(x, w):
    """
    :param x: ciag danych wejsciowych, NxM
    :param w: parametry modelu, KxM
    :return: wektor klas przewidzianych dla tych danych, Nx1
    """
    P = softmax(x @ w.transpose())
    return np.reshape(np.argmax(P, axis=1), (np.shape(x)[0], 1))


def model_selection(x_train, y_train, x_val, y_val, epochs, etas, mini_batch, lambdas):
    """
    :param x_train: ciag treningowy wejscia, NxM
    :param y_train: ciad treningowy wyjscia, Nx1
    :param x_val: ciag walidacyjny wejscia, N1xM
    :param y_val: ciag walidacyjny wyjscia, N1x1
    :param epochs: ilosc iteracji dla algorytmu gradientu prostego
    :param etas: krok uczenia
    :param mini_batch: rozmiar mini_batcha
    :param lambdas: wartosci wspolczynnika regularyzacji do sprawdzenia
    :return: najlepsze parametry modelu
    """
    y_train_onehot = one_hot(y_train)
    F = np.empty((len(lambdas), len(etas)))
    M = np.shape(x_train)[1]
    K = np.shape(y_train_onehot)[1]
    w0 = np.zeros((K, M))
    w_values = np.empty((len(lambdas), len(etas), 36, 324))
    best_eta = 0
    best_lambda = 0

    for l in range(len(lambdas)):
        print("Lambda:", lambdas[l])
        for e in range(len(etas)):
            print("Eta:", etas[e])
            w = stochastic_gradient_descent(lambda ww, x, y: regularized_likelihood_function_gradient(ww, x, y,
                                            lambdas[l]), x_train, y_train_onehot, np.array(w0), epochs, etas[e], mini_batch)
            w_values[l, e] = w
            F[l, e] = measure(y_val, prediction(x_val, w))
            if F[l, e] > F[best_lambda, best_eta]:
                best_lambda = l
                best_eta = e
            print(F[l, e] / np.shape(y_val)[0])
    print("Best lambda: ", lambdas[best_lambda])
    print("Best eta: ", etas[best_eta])

    return w_values[best_lambda, best_eta]


def model():
    train_file = open('extracted_hog.pkl', 'rb')
    data = pkl.load(train_file)
    train_file.close()

    with open('aug_extracted_hog.pkl', 'rb') as f:
        data_aug = pkl.load(f)

    with open('aug_extracted_hog_2.pkl', 'rb') as f:
        data_aug_2 = pkl.load(f)

    X = np.concatenate((data[0], data_aug[0]))
    Y = np.concatenate((data[1], data_aug[1]))
    val_size = 6000
    train_size = 40000
    # order = np.arange(np.shape(X)[0] - 5000)
    # shuffle(order)
    X_train = X[val_size:val_size + train_size]
    Y_train = Y[val_size:val_size + train_size]
    # X_train = X_train[order]
    # Y_train = Y_train[order]
    X_val = X[:val_size]
    Y_val = Y[:val_size]
    etas = [0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.03, 0.04]
    lambdas = [0, 0.0001, 0.001]  # for more than 100 epochs remove 0.01 lambda value - overflow
    w = model_selection(X_train, Y_train, X_val, Y_val, 100, etas, 50, lambdas)

    data_file = open('data.pkl', 'wb')
    pkl.dump(w, data_file)
    data_file.close()


def extract_to_file_hog(source, dest):
    train_file = open(source, 'rb')
    data = pkl.load(train_file)
    train_file.close()

    X = data[0]
    X_hog = hog_extraction(X)
    w = X_hog, data[1]

    hog_file = open(dest, 'wb')
    pkl.dump(w, hog_file)
    hog_file.close()


def test():
    train_file = open('train.pkl', 'rb')
    data = pkl.load(train_file)
    train_file.close()

    X = data[0]
    Y = data[1]
    x_test = X[24000:]
    y_test = Y[24000:]
    y_pred = predict(x_test)
    test = measure(y_test, y_pred)

    print(test / np.shape(y_test)[0])


extract_to_file_hog('train.pkl', 'extracted_hog.pkl')
# extract_to_file_hog('augmented_data.pkl', 'aug_extracted_hog.pkl')
# extract_to_file_hog('augmented_data_2.pkl', 'aug_extracted_hog_2.pkl')
# model()
# test()




