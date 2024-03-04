import os
import random
import time
import numpy as np
import sklearn.metrics as metrics
import torch
from torch import save
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.optim import Adadelta

from tool.loss.smooth import smooth_focal_weight
from data_ import *
from backbone_ import *
from loader1 import my_dataset as my_dataset_10s_smote
from tool.loss.focalloss import FocalLoss
from backbone_ import classifier


# Adapted from: https://github.com/QiXi9409/Simultaneous_ECG_Heartbeat
s = "python3 {}".format(os.getcwd(), 'mit.py')
os.system(s)
eps = 10000 
seed = 42
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class model():
    def __int__(self):
        pass

    def train(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
        device_ids = [0, 1]
        self.classifier = classifier()
        get_paprams(self.classifier)
        get_paprams(self.classifier.base)

        set_seed(seed)
        data_set = my_dataset_10s_smote(train=True)
        set_seed(seed)
        data_set_test = my_dataset_10s_smote(train=False, test=True)
        set_seed(seed)
        data_set_eval = my_dataset_10s_smote(train=False, test=False, eval=True)

        batch = 300
        totoal_epoch = 100 

        set_seed(seed)
        data_loader = DataLoader(data_set, batch, shuffle=True, collate_fn=detection_collate)
        set_seed(seed)
        data_loader_test = DataLoader(data_set_test, batch, False, collate_fn=detection_collate)
        set_seed(seed)
        data_loader_eval = DataLoader(data_set_eval, batch, False, collate_fn=detection_collate)

        self.classifier = self.classifier.cuda()
        self.classifier = DataParallel(self.classifier, device_ids=device_ids)
        optim = Adadelta(self.classifier.parameters(), 0.1, 0.9, weight_decay=1e-5)
        self.cretion = smooth_focal_weight()
        self.classifier.apply(weights_init)

        start_time = time.time()
        count = 0
        epoch = -1
        min_eval_loss = 100000000
        while epoch < eps:
            epoch += 1
            runing_losss = [0] * 5
            for data in data_loader:
                loss = [0] * 5
                y = data[1].cuda()
                x = data[0].cuda()
                optim.zero_grad()

                weight = torch.Tensor([0.5, 2, 0.5, 2]).cuda()
                predict = self.classifier(x)

                for i in range(5):
                    loss[i] = self.cretion(predict[i], y, weight)
                tmp = sum(loss)

                tmp.backward()
                optim.step()
                for i in range(5):
                    runing_losss[i] += (tmp.item())

                count += 1
            end_time = time.time()
            print(
                "epoch:{a}: loss:{b} spend_time:{c} time:{d}".format(a=epoch, b=sum(runing_losss),
                                                                     c=int(end_time - start_time),
                                                                     d=time.asctime()))
            start_time = end_time

            self.classifier.eval() 
            eval_loss = self.evaluation(self.classifier, data_loader_eval)

            if eval_loss < min_eval_loss:
                save(self.classifier.module.base.state_dict(), 'base_max.p')
                min_eval_loss = eval_loss

                self.test(self.classifier, data_loader_test)

            self.classifier.train() 
            if epoch % 10 == 0:
                adjust_learning_rate(optim, 0.9, epoch, totoal_epoch, 0.1)


    def evaluation(self, classifier, data_loader_eval):
        classifier.eval()
        all_predict = [[], [], [], [], []]
        all_ground = []
        runing_losss = [0] * 5
        with torch.no_grad():
            for data in data_loader_eval:
                loss = [0] * 5
                y = data[1].cuda()
                x = data[0].cuda()
                weight = torch.Tensor([0.5, 2, 0.5, 2]).cuda()

                predict_list = classifier(x)
                for i in range(5):
                    loss[i] = self.cretion(predict_list[i], y, weight)
                    predict, index = torch.max(predict_list[i], 1)
                    all_predict[i].extend(index.tolist())
                tmp = sum(loss)
                for i in range(5):
                    runing_losss[i] += (tmp.item())

                all_ground.extend(y.tolist())

        acc = metrics.accuracy_score(all_ground, all_predict[0])
        acc = np.round(acc * 100, 3)
        pre = metrics.precision_score(all_ground, all_predict[0], average=None)
        pre = np.round(pre * 100, 3)
        re = metrics.recall_score(all_ground, all_predict[0], average=None)
        re = np.round(re * 100, 3)
        f1 = metrics.f1_score(all_ground, all_predict[0], average=None)
        f1 = np.round(f1 * 100, 3)
        
        return sum(runing_losss)


    def test(self, classifier, data_loader_test):
        classifier.eval()
        all_predict = []
        all_ground = []

        with torch.no_grad():
            for data in data_loader_test:
                y = data[1].cuda()
                x = data[0].cuda()
                every_len = data[2]
                max_len = data[3]
                predict = classifier(x, every_len, max_len)[0]
                predict, index = torch.max(predict, 1)

                all_predict.extend(list(index.cpu().numpy()))
                all_ground.extend(list(y.cpu().numpy()))

        acc = metrics.accuracy_score(all_ground, all_predict)
        acc = np.round(acc * 100, 3)
        pre = metrics.precision_score(all_ground, all_predict, average=None)
        pre = np.round(pre * 100, 3)
        re = metrics.recall_score(all_ground, all_predict, average=None)
        re = np.round(re * 100, 3)
        f1 = metrics.f1_score(all_ground, all_predict, average=None)
        f1 = np.round(f1 * 100, 3)


if __name__ == '__main__':
    a = model()
    a.train()