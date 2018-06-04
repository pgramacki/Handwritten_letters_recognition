import Augmentor
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


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


def generate(x, y, window_size, magnitude, N):
    x_train = np.reshape(x, (-1, 56, 56))
    p = Augmentor.Pipeline()
    p.random_distortion(0.9, window_size, window_size, magnitude)
    g = p.keras_generator_from_array(x_train, y, batch_size=N)
    return next(g)


def perform_augmentation(N, window_size, magnitude):
    f = open('train.pkl', 'rb')
    data = pkl.load(f)
    f.close()

    x_train = data[0]
    y_train = data[1]
    x_aug, y_aug = generate(x_train, y_train, window_size, magnitude, N)
    x_aug = np.squeeze(x_aug)
    res = (x_aug, y_aug)
    print(np.shape(x_aug))
    print(np.shape(y_aug))

    with open('augmented_data_2.pkl', 'wb') as file:
        pkl.dump(res, file)


perform_augmentation(100000, 2, 1)
