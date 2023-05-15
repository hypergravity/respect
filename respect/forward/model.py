import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from astropy.time import Time
from ..utils import make_dataloader


class ForwardRespect(nn.Module):
    def __init__(self, ndim_in=3, ndim_out=1500, nhidden_syn=100, nhidden_cal=100, device="cpu"):
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

        # device
        self.device = device

        # model layers
        self.bn = nn.BatchNorm1d(ndim_in)

        self.syn = nn.Sequential(
            nn.Linear(ndim_in, nhidden_syn),
            nn.BatchNorm1d(nhidden_syn),
            nn.Tanh(),
            nn.Linear(nhidden_syn, ndim_out),
        )

        self.cal = nn.Sequential(
            nn.Linear(ndim_in, nhidden_cal),
            nn.BatchNorm1d(nhidden_cal),
            nn.Tanh(),
            nn.Linear(nhidden_cal, ndim_out),
        )
        self.to(self.device)

    def pred_syn(self, label):
        return self.syn(self.bn(label))

    def pred_cal(self, label):
        return self.cal(self.bn(label))

    def forward(self, label):
        x = self.bn(label)
        return self.syn(x) + self.cal(x)

    def set_data(self, label_syn=None, flux_syn=None, label_cal=None, flux_cal=None, batch_size=100):
        # automatically determine training mode
        self.train_syn = flux_syn is not None and label_syn is not None
        self.train_cal = flux_cal is not None and label_cal is not None
        assert self.train_syn or self.train_cal
        self.syn[-1].bias.data = torch.from_numpy(np.median(flux_syn, axis=0).astype(np.float32))

        if self.train_syn:
            self.dl_syn_train, self.dl_syn_test = make_dataloader(
                label_syn, flux_syn, batch_size=batch_size, device=self.device)
        if self.train_cal:
            self.dl_cal_train, self.dl_cal_test = make_dataloader(
                label_cal, flux_cal, batch_size=batch_size, device=self.device)

    def fit(self, lr_syn=1e-4, lr_cal=1e-3, wd_cal=1e-4, n_epoch=100, valid_step=10, plot=True, L1=True):

        # record batch / training / test loss
        self.batch_epoch = []
        self.valid_epoch = []
        self.batch_loss_syn = []
        self.batch_loss_cal = []
        self.train_loss_syn = []
        self.train_loss_cal = []
        self.test_loss_syn = []
        self.test_loss_cal = []

        # construct loss functions
        loss_syn = nn.MSELoss()
        loss_cal = nn.L1Loss() if L1 else nn.MSELoss()

        # construct optimizer
        params = []
        if self.train_syn:
            params.append({"params": self.bn.parameters(), "lr": lr_syn})
            params.append({"params": self.syn.parameters(), "lr": lr_syn})
        if self.train_cal:
            params.append({"params": self.cal.parameters(), "lr": lr_cal, "weight_decay": wd_cal})
        optimizer = torch.optim.RAdam(params)

        # train model
        self.train()
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
                    optimizer.step()
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
                    optimizer.step()
                batch_loss_cal = batch_loss_cal / batch_count_cal
                self.batch_loss_cal.append(batch_loss_cal)

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

            if i_epoch % valid_step == 0:
                prt_str = f"[Epoch {i_epoch:05d}/{n_epoch:05d}] - {Time.now().isot}:\n"
                if self.train_syn:
                    prt_str += f" => batch_loss_syn={self.batch_loss_syn[-1]}, train_loss_syn={self.train_loss_syn[-1]}, test_loss_syn={self.test_loss_syn[-1]} \n"
                if self.train_cal:
                    prt_str += f" => batch_loss_cal={self.batch_loss_cal[-1]}, train_loss_cal={self.train_loss_cal[-1]}, test_loss_cal={self.test_loss_cal[-1]} \n"
                print(prt_str)

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

    # def train_cal(self, flux_emp, label_emp):
    #     pass
    #
    # def train_both(self, flux_syn, label_syn, flux_emp, label_emp):
    #     pass
