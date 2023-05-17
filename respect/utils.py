import numpy as np
import torch
import torch.utils.data


def make_dataloader(*args, f_train=.9, batch_size=100, device=None):
    """
    dl_train, dl_test = make_dataloader(np.arange(100), f_train=0.9, batch_size=10)
    """
    if device is None:
        args = [torch.from_numpy(np.asarray(_, dtype=np.float32)) for _ in args]
    else:
        args = [torch.from_numpy(np.asarray(_, dtype=np.float32)).to(device=device) for _ in args]

    # split
    n_sample = args[0].shape[0]
    n_train = int(f_train * n_sample)
    # make dataset
    ds = torch.utils.data.TensorDataset(*args)
    # train valid random split
    ds_train, ds_test = torch.utils.data.random_split(ds, [n_train, n_sample - n_train])
    # make dataloader for training set
    dl_train = torch.utils.data.DataLoader(ds_train, sampler=None, batch_size=batch_size, shuffle=True)
    dl_test = torch.utils.data.DataLoader(ds_test, sampler=None, batch_size=batch_size, shuffle=True)
    return dl_train, dl_test
