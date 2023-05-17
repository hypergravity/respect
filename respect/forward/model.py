import os.path
import tempfile
from typing import Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from astropy.time import Time
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..utils import make_dataloader


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0., std=0.01)
        nn.init.zeros_(m.bias)


class ForwardRespect(nn.Module):
    def __init__(self,
                 ndim_in: int = 3,
                 ndim_out: int = 1500,
                 nhidden_syn: Tuple = (100, 200),
                 nhidden_cal: Tuple = (100, 200),
                 device: Union[str, int] = "cpu",
                 minit: bool = True,
                 ):
        """Forward RESPECT model

        Parameters
        ----------
        ndim_in: int
            number of label dimensions
        ndim_out: int
            number of flux dimensions
        nhidden_syn: tuple
            number of hidden neurons of syn network
        nhidden_cal: tuple
            number of hidden neurons of cal network
        device: str or int
            The device for training.
        minit: bool
            If True, manually initialize parameters.

        """
        super().__init__()
        self.batch_epoch = []
        self.valid_epoch = []
        self.batch_loss_syn = []
        self.batch_loss_cal = []
        self.train_loss_syn = []
        self.train_loss_cal = []
        self.test_loss_syn = []
        self.test_loss_cal = []

        # data loaders
        self.dl_syn_train = None
        self.dl_syn_test = None
        self.dl_cal_train = None
        self.dl_cal_test = None

        # train mode
        self.train_syn = False
        self.train_cal = False

        # _device
        if device != "cpu":
            assert torch.cuda.is_available()
            self._device = torch.device(device)

        # model layers
        self.bn = nn.BatchNorm1d(ndim_in)

        # syn network
        syn_layers = [
            nn.BatchNorm1d(ndim_in),
            nn.Linear(ndim_in, nhidden_syn[0]),
        ]
        for i_layer in range(len(nhidden_syn)):
            syn_layers.append(nn.BatchNorm1d(nhidden_syn[i_layer]))
            syn_layers.append(nn.Tanh())
            if i_layer == len(nhidden_syn) - 1:
                syn_layers.append(nn.Linear(nhidden_syn[i_layer], ndim_out))
            else:
                syn_layers.append(nn.Linear(nhidden_syn[i_layer], nhidden_syn[i_layer + 1]))
        self.syn = nn.Sequential(*syn_layers)

        # cal network
        cal_layers = [
            nn.BatchNorm1d(ndim_in),
            nn.Linear(ndim_in, nhidden_cal[0]),
        ]
        for i_layer in range(len(nhidden_cal)):
            cal_layers.append(nn.BatchNorm1d(nhidden_cal[i_layer]))
            cal_layers.append(nn.Tanh())
            if i_layer == len(nhidden_cal) - 1:
                cal_layers.append(nn.Linear(nhidden_cal[i_layer], ndim_out))
            else:
                cal_layers.append(nn.Linear(nhidden_cal[i_layer], nhidden_cal[i_layer + 1]))
        self.cal = nn.Sequential(*cal_layers)

        if minit:
            # init parameters
            self.apply(init_constant)

        # best state dict path
        self.temp_sd = os.path.join(tempfile.gettempdir(), "sd-best.joblib")

    @property
    def get_best_sd(self):
        return joblib.load(self.temp_sd)

    def pred_syn(self, label):
        return self.syn(self.bn(label))

    def pred_cal(self, label):
        return self.cal(self.bn(label))

    def forward(self, label):
        x = self.bn(label)
        return self.syn(x) + self.cal(x)

    def set_data(self, label_syn=None, flux_syn=None, label_cal=None, flux_cal=None, batch_size=128):

        if label_syn is not None and flux_syn is not None:
            self.dl_syn_train, self.dl_syn_test = make_dataloader(
                label_syn, flux_syn, batch_size=batch_size, device=self._device)
            # automatically determine the bias in last layer
            self.syn[-1].bias.data = torch.from_numpy(np.median(flux_syn, axis=0).astype(np.float32)).to(self._device)

        if label_cal is not None and flux_cal is not None:
            self.dl_cal_train, self.dl_cal_test = make_dataloader(
                label_cal, flux_cal, batch_size=batch_size, device=self._device)
            self.cal[-1].bias.data = torch.zeros_like(self.cal[-1].bias.data).to(self._device)

    def fit(self, lr_syn=(1e-3, 1e-6), lr_cal=(1e-3, 1e-6),
            wd_cal=1e-4, n_epoch=100, valid_step=10, plot=True, L1=True):
        if lr_syn is not None:
            self.train_syn = True
            lr_syn_0, lr_syn_1 = lr_syn
        else:
            self.train_syn = False
        if lr_cal is not None:
            self.train_cal = True
            lr_cal_0, lr_cal_1 = lr_cal
        else:
            self.train_cal = False

        # record batch / training / test loss
        self.batch_epoch = []
        self.valid_epoch = []
        self.batch_loss_syn = []
        self.batch_loss_cal = []
        self.train_loss_syn = []
        self.train_loss_cal = []
        self.test_loss_syn = []
        self.test_loss_cal = []

        self.to(self._device)
        # construct loss functions
        loss_syn = nn.MSELoss()
        loss_cal = nn.L1Loss() if L1 else nn.MSELoss()

        # construct optimizer
        if self.train_syn:
            optimizer_syn = torch.optim.RAdam(
                [
                    {"params": self.bn.parameters()},
                    {"params": self.syn.parameters()},
                ], lr=lr_syn_0
            )
            scheduler_syn = CosineAnnealingLR(optimizer_syn, T_max=n_epoch, eta_min=lr_syn_1)
        if self.train_cal:
            optimizer_cal = torch.optim.RAdam(self.cal.parameters(), lr=lr_cal_0, weight_decay=wd_cal)
            scheduler_cal = CosineAnnealingLR(optimizer_cal, T_max=n_epoch, eta_min=lr_cal_1)

        # train model
        test_loss_best = np.inf
        self.train()
        t0 = Time.now()
        t0_epoch = -1
        for i_epoch in range(n_epoch):
            # in each epoch
            self.batch_epoch.append(i_epoch)

            # train syn
            if self.train_syn:
                batch_count_syn = 0
                batch_loss_syn = 0
                for batch_label_syn, batch_flux_syn in self.dl_syn_train:
                    # for each batch
                    self.zero_grad()
                    batch_flux_syn_pred = self.pred_syn(batch_label_syn)
                    batch_loss_syn = loss_syn(batch_flux_syn_pred, batch_flux_syn)
                    batch_count_syn += len(batch_label_syn)
                    batch_loss_syn += batch_loss_syn.detach() * len(batch_label_syn)
                    # backward propagation
                    batch_loss_syn.backward()
                    optimizer_syn.step()
                batch_loss_syn = batch_loss_syn / batch_count_syn
                self.batch_loss_syn.append(batch_loss_syn)

            # train cal
            if self.train_cal:
                batch_count_cal = 0
                batch_loss_cal = 0
                for batch_label_cal, batch_flux_cal in self.dl_cal_train:
                    # for each batch
                    self.zero_grad()
                    batch_flux_cal_pred = self.forward(batch_label_cal)
                    batch_loss_cal = loss_cal(batch_flux_cal_pred, batch_flux_cal)
                    batch_count_cal += len(batch_label_cal)
                    batch_loss_cal += batch_loss_cal.detach() * len(batch_label_cal)
                    # backward propagation
                    batch_loss_cal.backward()
                    optimizer_cal.step()
                batch_loss_cal = batch_loss_cal / batch_count_cal
                self.batch_loss_cal.append(batch_loss_cal)

            if i_epoch % valid_step == 0:

                # validate model
                with torch.no_grad():
                    self.valid_epoch.append(i_epoch)

                    # train loss - syn
                    if self.train_syn:
                        train_count_syn = 0
                        train_loss_syn = 0
                        for batch_label_syn, batch_flux_syn in self.dl_syn_train:
                            batch_flux_syn_pred = self.pred_syn(batch_label_syn)
                            batch_loss_syn = loss_syn(batch_flux_syn_pred, batch_flux_syn)
                            train_count_syn += len(batch_label_syn)
                            train_loss_syn += batch_loss_syn.detach() * len(batch_label_syn)
                        train_loss_syn /= train_count_syn
                        self.train_loss_syn.append(train_loss_syn)

                    # train loss - cal
                    if self.train_cal:
                        train_count_cal = 0
                        train_loss_cal = 0
                        for batch_label_cal, batch_flux_cal in self.dl_cal_train:
                            batch_flux_cal_pred = self.forward(batch_label_cal)
                            batch_loss_cal = loss_cal(batch_flux_cal_pred, batch_flux_cal)
                            train_count_cal += len(batch_label_cal)
                            train_loss_cal += batch_loss_cal.detach() * len(batch_label_cal)
                        train_loss_cal /= train_count_cal
                        self.train_loss_cal.append(train_loss_cal)

                    # test loss - syn
                    if self.train_syn:
                        test_count_syn = 0
                        test_loss_syn = 0
                        for batch_label_syn, batch_flux_syn in self.dl_syn_test:
                            batch_flux_syn_pred = self.pred_syn(batch_label_syn)
                            batch_loss_syn = loss_syn(batch_flux_syn_pred, batch_flux_syn)
                            test_count_syn += len(batch_label_syn)
                            test_loss_syn += batch_loss_syn.detach() * len(batch_label_syn)
                        test_loss_syn /= test_count_syn
                        self.test_loss_syn.append(test_loss_syn)

                    # test loss - cal
                    if self.train_cal:
                        test_count_cal = 0
                        test_loss_cal = 0
                        for batch_label_cal, batch_flux_cal in self.dl_cal_test:
                            batch_flux_cal_pred = self.forward(batch_label_cal)
                            batch_loss_cal = loss_cal(batch_flux_cal_pred, batch_flux_cal)
                            test_count_cal += len(batch_label_cal)
                            test_loss_cal += batch_loss_cal.detach() * len(batch_label_cal)
                        test_loss_cal /= test_count_cal
                        self.test_loss_cal.append(test_loss_cal)

                    if self.train_cal:
                        test_loss_current = float(test_loss_cal.cpu())
                    else:
                        test_loss_current = float(test_loss_syn.cpu())

                    save = test_loss_current < test_loss_best
                    if save:
                        # save state dict
                        joblib.dump(self.state_dict(), self.temp_sd)
                t1 = Time.now()
                t1_epoch = i_epoch

                prt_str = f"[Epoch {i_epoch:05d}/{n_epoch:05d} | {(t1 - t0).sec / (t1_epoch - t0_epoch):.1f} s/epoch]" \
                          f" - {t1.isot} - save={save}" \
                          f" - test_loss_best={test_loss_best:.7f}" \
                          f" - test_loss_current={test_loss_current:.7f}:\n"
                if self.train_syn:
                    prt_str += f" => (lr_syn={scheduler_syn.get_last_lr()[0]:.1e})" \
                               f" batch_loss_syn={self.batch_loss_syn[-1]:.7f}," \
                               f" train_loss_syn={self.train_loss_syn[-1]:.7f}," \
                               f" test_loss_syn={self.test_loss_syn[-1]:.7f} \n"
                if self.train_cal:
                    prt_str += f" => (lr_cal={scheduler_cal.get_last_lr()[0]:.1e})" \
                               f" batch_loss_cal={self.batch_loss_cal[-1]:.7f}," \
                               f" train_loss_cal={self.train_loss_cal[-1]:.7f}," \
                               f" test_loss_cal={self.test_loss_cal[-1]:.7f} \n"
                print(prt_str)

                # iterate best loss
                test_loss_best = test_loss_current
                # iterate tic time
                t0 = t1
                t0_epoch = t1_epoch

            if self.train_syn:
                scheduler_syn.step()
            if self.train_cal:
                scheduler_cal.step()

        # restore best state dict
        self.load_state_dict(joblib.load(self.temp_sd))
        self.train_loss_syn = [float(_.cpu()) for _ in self.train_loss_syn]
        self.train_loss_cal = [float(_.cpu()) for _ in self.train_loss_cal]
        self.test_loss_syn = [float(_.cpu()) for _ in self.test_loss_syn]
        self.test_loss_cal = [float(_.cpu()) for _ in self.test_loss_cal]

        if plot:
            lw = 2
            plt.figure(figsize=(6, 4))
            plt.plot(np.log10(self.train_loss_syn), "-", color="tab:blue", lw=lw, label="training loss [syn]")
            plt.plot(np.log10(self.train_loss_cal), "--", color="tab:blue", lw=lw, label="training loss [cal]")
            plt.plot(np.log10(self.test_loss_syn), "-", color="tab:red", lw=lw, label="test loss [syn]")
            plt.plot(np.log10(self.test_loss_cal), "--", color="tab:red", lw=lw, label="test loss [cal]")
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("log10(loss)")

        # switch to evaluation mode
        self.eval()
