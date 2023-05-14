import numpy as np


def data_loader(*args, f_train=0.7, batch_size=50, train=True):
    n_data = len(args[0])
    n_train = int(n_data * f_train)

    # shuffle data
    index = np.arange(0, n_data, dtype=int)
    # np.random.seed(0)
    np.random.shuffle(index)

    if train:
        # get training set
        for i in range(0, n_train, batch_size):
            yield (_[index[i:min(i + batch_size, n_train)]] for _ in args)
    else:
        # get test set
        for i in range(n_train, n_data, batch_size):
            yield (_[index[i:min(i + batch_size, n_data)]] for _ in args)
