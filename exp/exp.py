import os
from time import time

import numpy as np
import torch
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import MyDataset
from model.GDN import GDN
from utils.earlystoping import EarlyStopping
from utils.evaluate import evaluate, myevaluate
from utils.getdata import get_data


class Exp:
    def __init__(self, config):
        self.__dict__.update(config)
        self._get_data()
        self._get_model()

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _get_data(self):
        data = get_data(self.dataset, self.data_dir)
        if self.verbose:
            for k, v in data.items():
                print(k, ': ', v.shape)

        self.n_feature = data['train_data'].shape[-1]

        trainset = MyDataset(data['train_data'], data['train_label'], self.windows_size)
        validset = MyDataset(data['valid_data'], data['valid_label'], self.windows_size)
        threset = MyDataset(data['thre_data'], data['thre_label'], self.windows_size)
        testset = MyDataset(data['test_data'], data['test_label'], self.windows_size)

        self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.validloader = DataLoader(validset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.threloader = DataLoader(threset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def _get_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GDN(feature_num=self.n_feature,
                         timd_num=self.windows_size,
                         embed_dim=self.dim,
                         topk=self.topk
                         ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.75 ** ((epoch - 1) // 1))
        self.early_stopping = EarlyStopping(patience=self.patience,
                                            verbose=self.verbose,
                                            path=self.model_dir + self.dataset + '_model.pkl')
        self.criterion = nn.MSELoss(reduction="mean")
        print(self.device)

    def train(self):
        for e in range(self.epochs):
            start = time()

            self.model.train()
            train_loss = []
            for (batch_x, batch_y, _) in tqdm(self.trainloader):
                self.optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device).permute(0, 2, 1).contiguous()
                batch_y = batch_y.float().to(self.device)

                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                valid_loss = []
                for (batch_x, batch_y, _) in tqdm(self.validloader):
                    batch_x = batch_x.float().to(self.device).permute(0, 2, 1).contiguous()
                    batch_y = batch_y.float().to(self.device)

                    output = self.model(batch_x)
                    loss = self.criterion(output, batch_y)
                    valid_loss.append(loss.item())

            train_loss, valid_loss = np.average(train_loss), np.average(valid_loss)
            end = time()
            print("Epoch: {0} || Train Loss: {1:.6f} Valid Loss: {2:.6f} || Cost: {3:.6f}".format(
                e + 1, train_loss, valid_loss, end - start))

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                break
            self.scheduler.step()

        self.model.load_state_dict(torch.load(self.model_dir + self.dataset + '_model.pkl'))

    def test(self):
        # 异常得分为预测值与真实值的差异 （55个指标的最大值）
        self.model.load_state_dict(torch.load(self.model_dir + self.dataset + '_model.pkl'))
        self.criterion = nn.L1Loss(reduction="none")

        with torch.no_grad():
            self.model.eval()

            valid_score = []
            for (batch_x, batch_y, _) in tqdm(self.validloader):
                batch_x = batch_x.float().to(self.device).permute(0, 2, 1).contiguous()
                batch_y = batch_y.float().to(self.device)

                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)

                valid_score.append(loss.detach().cpu().numpy())

            thre_score, thre_label = [], []
            for (batch_x, batch_y, batch_label) in tqdm(self.threloader):
                batch_x = batch_x.float().to(self.device).permute(0, 2, 1).contiguous()
                batch_y = batch_y.float().to(self.device)

                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)

                thre_score.append(loss.detach().cpu().numpy())
                thre_label.append(batch_label.detach().cpu().numpy())

            test_score, test_label = [], []
            for (batch_x, batch_y, batch_label) in tqdm(self.testloader):
                batch_x = batch_x.float().to(self.device).permute(0, 2, 1).contiguous()
                batch_y = batch_y.float().to(self.device)

                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)

                test_score.append(loss.detach().cpu().numpy())
                test_label.append(batch_label.detach().cpu().numpy())

        valid_score = np.concatenate(valid_score, axis=0)
        thre_score = np.concatenate(thre_score, axis=0)
        test_score = np.concatenate(test_score, axis=0)

        thre_label = np.concatenate(thre_label, axis=0)
        test_label = np.concatenate(test_label, axis=0)

        precision, precision_adjust, recall, recall_adjust, f_score, f_score_adjust, auc = evaluate(valid_score,
                                                                                                    test_score,
                                                                                                    test_label)

        print("Paper   || P : {0:.4f} | R : {1:.4f} | F1 : {2:.4f} | AUC : {3:.4f}".format(precision, recall, f_score,
                                                                                           auc))
        print("Paper+  || P : {0:.4f} | R : {1:.4f} | F1 : {2:.4f} | AUC : {3:.4f}".format(precision_adjust,
                                                                                           recall_adjust,
                                                                                           f_score_adjust, auc))

        precision, precision_adjust, recall, recall_adjust, f_score, f_score_adjust, auc = myevaluate(thre_score,
                                                                                                      thre_label,
                                                                                                      test_score,
                                                                                                      test_label)

        print("search+ || P : {0:.4f} | R : {1:.4f} | F1 : {2:.4f} | AUC : {3:.4f}".format(precision, recall, f_score,
                                                                                           auc))
        print("Search+ || P : {0:.4f} | R : {1:.4f} | F1 : {2:.4f} | AUC : {3:.4f}".format(precision_adjust,
                                                                                           recall_adjust,
                                                                                           f_score_adjust, auc))
