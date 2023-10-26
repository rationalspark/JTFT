# This file is licensed under the GNU General Public License (GPL) version 2.0.
# See the LICENSE file or https://www.gnu.org/licenses/gpl-2.0.html for more details.

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import JTFT
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import sys
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
import pandas as pd
import copy

warnings.filterwarnings('ignore')

class Exp_Main_JTFT(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main_JTFT, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'JTFT': JTFT,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, model):
        model_optim = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, verbose=False):
        total_loss = []
        if verbose:
            preds = []
            trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.args.use_mark:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                else:
                    batch_x_mark = None
                    batch_y_mark = None   
                outputs = self.model(batch_x, z_mark=batch_x_mark, target_mark=batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
                if verbose:
                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()
                    preds.append(pred.numpy())
                    trues.append(true.numpy())
        total_loss = np.average(total_loss)
        if verbose:
            preds = np.array(preds)
            trues = np.array(trues)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
            print('ms/ma/rse:{:.4f}, {:.4f}, {:.4f}'.format(mse, mae, rse))
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        train_steps = len(train_loader)
        print('Length of train/val/test loader', len(train_loader), len(vali_loader), len(test_loader))
            
        criterion = self._select_criterion()
        if self.args.use_huber_loss:
            print("Use huber loss for train and validation, test loss remains MSE")
            criterion_huber = nn.HuberLoss(delta=self.args.huber_delta)
        if not self.args.ini_with_low_freq:
            #Calculate the initial frequencies
            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                x = batch_x[:, -self.args.seq_len:, :].float().to(self.device)
                #Accumulate amplitude for frequecies
                if not self.args.use_multi_gpu:
                    self.model.model.accum_freq_amp(x)
                else:
                    self.model.module.model.accum_freq_amp(x)
            #Obtain initial frequencies
            if not self.args.use_multi_gpu:
                self.model.model.comp_ini_freq()
            else:
                self.model.module.model.comp_ini_freq()
        #Training
        model_optim = self._select_optimizer(self.model)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                        steps_per_epoch = train_steps,
                        pct_start = self.args.pct_start,
                        epochs = self.args.train_epochs,
                        max_lr = self.args.learning_rate)
        if self.args.resume_after_epo != 0:
            print('loading model')
            self.model.load_state_dict(torch.load(path + '/checkpoint.pth'))
            for epoch in range(self.args.resume_after_epo):
                if self.args.lradj != 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
                else:
                    for step in range(len(train_loader)):
                        adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                        scheduler.step()
                
        #Training models, return the best result obtained on 'ctrl+c'
        try:
            for epoch in range(self.args.resume_after_epo, self.args.train_epochs):
                iter_count = 0
                train_loss = []
                self.model.train()
                epoch_time = time.time()
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    if self.args.use_mark:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)
                    else:
                        batch_x_mark = None
                        batch_y_mark = None    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = self.model(batch_x, z_mark=batch_x_mark, target_mark=batch_y_mark)
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    if self.args.use_huber_loss:
                        loss = criterion_huber(outputs, batch_y)
                    else:
                        loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                    loss.backward()
                    model_optim.step()
                    if self.args.lradj == 'TST':
                        adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                        scheduler.step()
                
                val_time=time.time()
                print("Epoch: {} cost time: {:.2f}".format(epoch + 1, val_time - epoch_time), end=" ")
                train_loss = np.average(train_loss)
                if self.args.use_huber_loss:
                    vali_loss = self.vali(vali_data, vali_loader, criterion_huber)
                else:
                    vali_loss = self.vali(vali_data, vali_loader, criterion)
                verbose_test=False
                if epoch >= self.args.min_epochs:
                    early_stopping(vali_loss, self.model, path)
                    if early_stopping.counter==0:
                        verbose_test=True
                test_loss = self.vali(test_data, test_loader, criterion, verbose=verbose_test)
                print("batchs {}, val/test time {:.2f}".format(i, time.time() - val_time))
                print("Epoch: {}, Steps: {} | Train {:,.5f} Vali Loss: {:.5f} Test Loss: {:.5f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                if not self.args.use_multi_gpu:
                    self.model.model.show_freqs(n_disp=self.args.n_freq)
                else:
                    self.model.module.model.show_freqs(n_disp=self.args.n_freq)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                if self.args.lradj != 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
                else:
                    print('Updating learning rate to {}'.format(model_optim.state_dict()['param_groups'][0]['lr']))

        except KeyboardInterrupt:
            print("KeyboardInterrupt, return the current best model")
        self.model.load_state_dict(torch.load(path +'/checkpoint.pth', map_location=self.device))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            path = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(torch.load(path + '/checkpoint.pth'))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.args.use_mark:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                else:
                    batch_x_mark = None
                    batch_y_mark = None    
                outputs = self.model(batch_x, z_mark=batch_x_mark, target_mark=batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                pred = outputs
                true = batch_y 
                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('Test_loss(pl {}): mse:{:.5f}, mae:{:.5f}, rse:{:.5f}'.format(self.args.pred_len, mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + " \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()
        np.save(folder_path + 'pred.npy', preds)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                outputs = self.model(batch_x)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

