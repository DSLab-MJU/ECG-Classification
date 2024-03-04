import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
import os
import random
import torch
import torch.nn.functional as F
from torch.optim import Adam, Adadelta
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import minmax_scale

from model import Transformer
from optim import ScheduledOptim
from dataset import SignalDataset
from config2 import *
from FocalLoss import FocalLoss
from entropy import *
from roc import plot_roc
from data_loader import loader
from config import cfg
from detection_collate_mit import call_back
from base_process import base_process
from backbone_ import *
from RCN import ROI as roi
from pre import pre as pre


# Adapted from: https://github.com/QiXi9409/Simultaneous_ECG_Heartbeat
# https://github.com/yangenshen/FusingTransformerModelwithTemporalFeaturesforECGHeartbeatClassification
FL = FocalLoss(class_num=5, gamma=1.5, average=False)
sys.path.append(os.getcwd())
eps = 10000
fe_size = 32 
ent_num = '02'
seed = 42
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def norm_data2(data):
    df = pd.DataFrame(data)
    df = df.transpose()
    norm = minmax_scale(df)
    norm_data = norm.transpose()
    return norm_data


def cal_loss(pred, label, device):
    cnt_per_class = np.zeros(5)
    loss = FL(pred, label, device)

    loss = F.cross_entropy(pred, label, reduction='sum')
    pred = pred.max(1)[1]
    n_correct = pred.eq(label).sum().item()
    cnt_per_class = [cnt_per_class[j] + pred.eq(j).sum().item() for j in range(class_num)]
    return loss, n_correct, cnt_per_class

def cal_statistic(cm):
    total_pred = cm.sum(0)
    total_true = cm.sum(1)

    acc_SP = sum([cm[i, i] for i in range(class_num)]) / total_pred[:class_num].sum()
    pre_i = [cm[i, i] / total_pred[i] for i in range(class_num)]
    rec_i = [cm[i, i] / total_true[i] for i in range(class_num)]
    F1_i = [2 * pre_i[i] * rec_i[i] / (pre_i[i] + rec_i[i]) for i in range(class_num)]

    pre_i = np.array(pre_i)
    rec_i = np.array(rec_i)
    F1_i = np.array(F1_i)
    pre_i[np.isnan(pre_i)] = 0
    rec_i[np.isnan(rec_i)] = 0
    F1_i[np.isnan(F1_i)] = 0

    return acc_SP, list(pre_i), list(rec_i), list(F1_i)


def train_epoch(train_loader, device, model, optimizer, total_num):
    all_labels = []
    all_res = []
    model.train()
    total_loss = 0
    total_correct = 0
    cnt_per_class = np.zeros(class_num)
    for batch in tqdm(train_loader, mininterval=0.5, desc='- (Training)  ', leave=False):
        sig, label = map(lambda x: x.to(device), batch)

        optimizer.zero_grad()
        pred = model(sig)
        all_labels.extend(label.cpu().numpy())
        all_res.extend(pred.max(1)[1].cpu().numpy())

        loss, n_correct, cnt = cal_loss(pred, label, device)
        loss.backward()

        optimizer.step_and_update_lr()

        total_loss += loss.item()
        total_correct += n_correct
        cnt_per_class += cnt
    train_loss = total_loss / total_num
    train_acc = total_correct / total_num
    return train_loss, train_acc, cnt_per_class


def eval_epoch(valid_loader, device, model, total_num):
    all_labels = []
    all_res = []
    model.eval()
    total_loss = 0
    total_correct = 0
    cnt_per_class = np.zeros(class_num)
    with torch.no_grad():
        for batch in tqdm(valid_loader, mininterval=0.5, desc='- (Validation)  ', leave=False):
            sig, label = map(lambda x: x.to(device), batch)

            pred = model(sig)  
            all_labels.extend(label.cpu().numpy())
            all_res.extend(pred.max(1)[1].cpu().numpy())
            loss, n_correct, cnt = cal_loss(pred, label, device)

            total_loss += loss.item()
            total_correct += n_correct
            cnt_per_class += cnt
    cm = confusion_matrix(all_labels, all_res)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    acc = accuracy_score(all_labels, all_res)
    acc = round(acc*100, 2)
    pre = precision_score(all_labels, all_res, average=None)
    pre = np.round(pre*100, 2)
    sen = recall_score(all_labels, all_res, average=None)
    sen = np.round(sen*100, 2)
    f1s = f1_score(all_labels, all_res, average=None)
    f1s = np.round(f1s*100, 2)

    valid_loss = total_loss / total_num
    valid_acc = total_correct / total_num
    return valid_loss, valid_acc, cnt_per_class, sum(rec_i[1:]) * 0.6 + sum(pre_i[1:]) * 0.4


def test_epoch(valid_loader, device, model, total_num):
    all_labels = []
    all_res = []
    all_pred = []
    model.eval()
    total_loss = 0
    total_correct = 0
    cnt_per_class = np.zeros(class_num)
    with torch.no_grad():
        for batch in tqdm(valid_loader, mininterval=0.5, desc='- (Validation)  ', leave=False):
            sig, label = map(lambda x: x.to(device), batch)

            pred = model(sig) 
            all_labels.extend(label.cpu().numpy())
            all_res.extend(pred.max(1)[1].cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
            loss, n_correct, cnt = cal_loss(pred, label, device)

            total_loss += loss.item()
            total_correct += n_correct
            cnt_per_class += cnt

    all_pred = np.array(all_pred)
    plot_roc(all_labels,all_pred)
    cm = confusion_matrix(all_labels, all_res)
    acc = accuracy_score(all_labels, all_res)
    acc = round(acc*100, 2)
    pre = precision_score(all_labels, all_res, average=None)
    pre = np.round(pre*100, 2)
    sen = recall_score(all_labels, all_res, average=None)
    sen = np.round(sen*100, 2)
    f1s = f1_score(all_labels, all_res, average=None)
    f1s = np.round(f1s*100, 2)

class model(base_process):
    def train_stage_2(self):
        batch = 240
        device_ids = [0, 1]

        set_seed(seed)
        data_set = loader(os.path.join(os.getcwd(), 'data_2'), {"mode": "training"}, sp=2)
        set_seed(seed)
        data_set_test = loader(os.path.join(os.getcwd(), 'data_2'),{"mode": "test"}, sp=2)
        set_seed(seed)
        data_set_eval = loader(os.path.join(os.getcwd(), 'data_2'),{"mode": "eval"}, sp=2)

        set_seed(seed)
        data_loader = DataLoader(data_set, batch, True, collate_fn=call_back.detection_collate_RPN)
        set_seed(seed)
        data_loader_test = DataLoader(data_set_test, batch, False, collate_fn=call_back.detection_collate_RPN)
        set_seed(seed)
        data_loader_eval = DataLoader(data_set_eval, batch, False, collate_fn=call_back.detection_collate_RPN)

        start_time = time.time()
        optim_a = Adadelta([{'params': self.pre.parameters()},
                            {'params': self.ROI.parameters()}], lr=0.15, weight_decay=1e-5)
        cfg.test = False
        count = 0
        train_loss_all, eval_loss_all = [], []
        min_eval_loss = 1000000000
        for epoch in range(eps):
            runing_losss = 0.0
            cls_loss = 0
            coor_loss = 0
            cls_loss2 = 0
            coor_loss2 = 0
            count += 1
            
            for data in data_loader:
                y = data[1]
                x = data[0].cuda()
                peak = data[2]
                num = data[3]
                optim_a.zero_grad()

                with torch.no_grad():
                    if self.flag >= 2:
                        result = self.base_process(x, y, peak)
                        feat1 = result['feat_8']
                        feat2 = result['feat_16']
                        feat3 = result['feat_32']
                        feat4 = result['feat_64']
                        label = result['label']
                        loss_box = result['loss_box']
                        cross_entropy = result['cross_entropy']

                cls_score = self.pre(feat1, feat2, feat3, feat4) 
                cls_score = self.ROI(cls_score) 

                cross_entropy2 = self.tool2.cal_loss2(cls_score, label) 

                loss_total = cross_entropy2
                loss_total.backward()
                optim_a.step()
                runing_losss += loss_total.item()
                cls_loss2 += cross_entropy2.item()
                cls_loss += cross_entropy.item()
                coor_loss += loss_box.item()
            end_time = time.time()
            torch.cuda.empty_cache()
            print(
                "epoch:{a} time:{ff}: loss:{b:.4f} cls:{d:.4f} cor{e:.4f} cls2:{f:.4f} cor2:{g:.4f} date:{fff}".format(
                    a=epoch,
                    b=runing_losss,
                    d=cls_loss,
                    e=coor_loss,
                    f=cls_loss2,
                    g=coor_loss2, ff=int(end_time - start_time),
                    fff=time.asctime()))
            train_loss_all.append(runing_losss)
            
            self.pre.eval()
            self.ROI.eval()
            eval_loss = self.pre_eval(self.pre, self.ROI, data_loader_eval)
            eval_loss_all.append(eval_loss)
            if eval_loss < min_eval_loss:
                min_eval_loss = eval_loss

                self.pre_eval(self.pre, self.ROI, data_loader_test)
                
            self.pre.train()
            self.ROI.train()
            
            p = None

            start_time = end_time

        train_data, train_label = [], []
        eval_data, eval_label = [], []
        test_data, test_label = [], []

        for data in data_loader:
            y = data[1]
            x = data[0].cuda()
            peak = data[2]

            with torch.no_grad():
                if self.flag >= 2:
                    result = self.base_process_2(x, y, peak)
                    data_ = result['x']
                    label = result['label']
                    loss_box = result['loss_box']
                    cross_entropy = result['cross_entropy']
                    train_data.extend(data_.cpu())
                    train_label.extend(label.cpu())

        for data in data_loader_eval:
            y = data[1]
            x = data[0].cuda()
            peak = data[2]

            with torch.no_grad():
                if self.flag >= 2:
                    result = self.base_process_2(x, y, peak)
                    data_ = result['x']
                    label = result['label']
                    loss_box = result['loss_box']
                    cross_entropy = result['cross_entropy']
                    eval_data.extend(data_.cpu())
                    eval_label.extend(label.cpu())

        for data in data_loader_test:
            y = data[1]
            x = data[0].cuda()
            peak = data[2]
            with torch.no_grad():
                if self.flag >= 2:
                    result = self.base_process_2(x, y, peak)
                    data_ = result['x']
                    label = result['label']
                    loss_box = result['loss_box']
                    cross_entropy = result['cross_entropy']
                    test_data.extend(data_.cpu())
                    test_label.extend(label.cpu())

        '''
        train_data = torch.stack(train_data, 0).numpy()
        train_label = torch.LongTensor(train_label).numpy()

        eval_data = torch.stack(eval_data, 0).numpy()
        eval_label = torch.LongTensor(eval_label).numpy()

        test_data = torch.stack(test_data, 0).numpy()
        test_label = torch.LongTensor(test_label).numpy()

        train_data = norm_data2(train_data)
        eval_data = norm_data2(eval_data)
        test_data = norm_data2(test_data)

        np.save('./final_train_data', train_data)
        np.save('./final_train_label', train_label)

        np.save('./final_test_data', test_data)
        np.save('./final_test_label', test_label)

        np.save('./final_eval_data', eval_data)
        np.save('./final_eval_label', eval_label)
        '''

        train_data = np.load('./final_diff_data.npy')
        train_label = np.load('./final_diff_label.npy')

        test_data = np.load('./final_test_data.npy')
        test_label = np.load('./final_test_label.npy')

        eval_data = np.load('./final_eval_data.npy')
        eval_label = np.load('./final_eval_label.npy')

        return train_data, train_label, test_data, test_label, eval_data, eval_label

    def base_process_2(self, x, y, peak): 
        cross_entropy, loss_box = torch.ones(1), torch.ones(1)
        with torch.no_grad():
            x1, x2, x3, x4 = self.features(x)
            if self.flag == 3:
                predict_confidence, box_predict = self.RPN(x1, x2, x3, x4)
                proposal, batch_offset, batch_conf = self.tool.get_proposal(predict_confidence, box_predict,
                                                                            y, test=True)

            proposal, label = self.tool2.pre_gt_match_uniform(proposal, y, training=True, params={'peak': peak})

            if 1:
                for i in range(len(proposal)):
                    tmp = torch.zeros(proposal[i].size()[0], 1).fill_(
                        i).cuda()
                    proposal[i] = torch.cat([tmp, proposal[i]], 1)
                proposal = torch.cat(proposal, 0)

            feat4, label, class_num = self.tool2.roi_pooling_cuda(x4, proposal, label=label, stride=64,
                                                                  pool=self.pool4,
                                                                  batch=True)
            feat3 = \
                self.tool2.roi_pooling_cuda(x3, proposal, stride=64, pool=self.pool3,
                                            batch=True, label=None)[0]
            feat2 = \
                self.tool2.roi_pooling_cuda(x2, proposal, stride=32,
                                            pool=self.pool2,
                                            batch=True, label=None)[0]
            feat1 = \
                self.tool2.roi_pooling_cuda(x1, proposal, stride=16,
                                            pool=self.pool1,
                                            batch=True, label=None, )[0]

            x = self.pre(feat1, feat2, feat3, feat4) 
            x = x.view(-1, fe_size * 15) 
            if self.flag == 2:
                result = {}
                result['x'] = x
                result['label'] = label
                result['predict_offset'] = 0
                result['class_num'] = class_num
                result['batch_cor_weight'] = 0
                result['cross_entropy'] = cross_entropy
                result['loss_box'] = loss_box
                return result
            elif self.flag == 3:
                result = {}
                result['x'] = x
                result['label'] = label
                result['class_num'] = class_num
                result['cross_entropy'] = cross_entropy
                result['loss_box'] = loss_box

                return result

    def pre_eval(self, pre, ROI, data_loader_eval):
        runing_losss = 0.0
        with torch.no_grad():
            for data in data_loader_eval:
                y = data[1]
                x = data[0].cuda()
                peak = data[2]
                num = data[3]
                
                if self.flag >= 2:
                    result = self.base_process(x, y, peak)
                    feat1 = result['feat_8']
                    feat2 = result['feat_16']
                    feat3 = result['feat_32']
                    feat4 = result['feat_64']
                    label = result['label']
                    loss_box = result['loss_box']
                    cross_entropy = result['cross_entropy']

            cls_score = pre(feat1, feat2, feat3, feat4)
            cls_score = ROI(cls_score)

            cross_entropy2 = self.tool2.cal_loss2(cls_score, label)
            loss_total = cross_entropy2
            runing_losss += loss_total.item()

        return runing_losss


if __name__ == '__main__':
    a = model()
    raw_train_data, raw_train_label, raw_test_data, raw_test_label, raw_valid_data, raw_valid_label = a.train_stage_2()

    model_name = 'transform.chkpt'

    raw_train_data = pd.DataFrame(raw_train_data)
    raw_train_data = raw_train_data.transpose()
    raw_train_data = minmax_scale(raw_train_data)
    raw_train_data = raw_train_data.transpose()
    raw_train_data = pd.DataFrame(raw_train_data) 
    raw_train_data = raw_train_data.values

    raw_test_data = pd.DataFrame(raw_test_data)
    raw_test_data = raw_test_data.transpose()
    raw_test_data = minmax_scale(raw_test_data)
    raw_test_data = raw_test_data.transpose()
    raw_test_data = pd.DataFrame(raw_test_data) 
    raw_test_data = raw_test_data.values

    raw_valid_data = pd.DataFrame(raw_valid_data)
    raw_valid_data = raw_valid_data.transpose()
    raw_valid_data = minmax_scale(raw_valid_data)
    raw_valid_data = raw_valid_data.transpose()
    raw_valid_data = pd.DataFrame(raw_valid_data) 
    raw_valid_data = raw_valid_data.values

    raw_train = np.c_[raw_train_data, raw_train_label]
    raw_test = np.c_[raw_test_data, raw_test_label]
    raw_valid = np.c_[raw_valid_data, raw_valid_label]


    cuda = True
    device = torch.device("cuda" if cuda else "cpu")

    train_data = SignalDataset(raw_train)
    valid_data = SignalDataset(raw_valid)
    test_data = SignalDataset(raw_test)
    train_loader = DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                num_workers=2,
                                shuffle=True)
    valid_loader = DataLoader(dataset=valid_data,
                                batch_size=batch_size,
                                num_workers=2,
                                shuffle=True)
    test_loader = DataLoader(dataset=test_data,
                                batch_size=batch_size,
                                num_workers=2,
                                shuffle=True)

    model = Transformer(device=device, d_feature=train_data.sig_len, d_model=d_model, d_inner=d_inner,
                        n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout, class_num=class_num).cuda()

    model = DataParallel(model, device_ids=[0, 1])

    optimizer = ScheduledOptim(
        Adam(filter(lambda x: x.requires_grad, model.parameters()),
                betas=(0.9, 0.98), eps=1e-09), d_model, warm_steps)
    train_accs = []
    valid_accs = []
    eva_indis = []
    train_losses = []
    valid_losses = []
    for epoch_i in range(epoch):
        print('[ Epoch', epoch_i, ']')
        start = time.time()
        train_loss, train_acc, cnt = train_epoch(train_loader, device, model, optimizer, train_data.__len__())
        print('  - (Training)  loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '
                'elapse: {elapse:3.3f} min'.format(loss=train_loss, accu=100 * train_acc,
                                                    elapse=(time.time() - start) / 60))
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        start = time.time()
        valid_loss, valid_acc, cnt, eva_indi = eval_epoch(valid_loader, device, model, valid_data.__len__())
        print('  - (Validation)  loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '
                'elapse: {elapse:3.3f} min'.format(loss=valid_loss, accu=100 * valid_acc,
                                                    elapse=(time.time() - start) / 60))
        valid_accs.append(valid_acc)
        eva_indis.append(eva_indi)
        valid_losses.append(valid_loss)
        model_state_dict = model.state_dict()

        checkpoint = {
            'model': model_state_dict,
            'config_file': 'config',
            'epoch': epoch_i}

        if eva_indi >= max(eva_indis):
            torch.save(checkpoint, model_name)

    test_model_name = model_name
    model = Transformer(device=device, d_feature=test_data.sig_len, d_model=d_model, d_inner=d_inner,
                        n_layers=num_layers, n_head=num_heads, d_k=64, d_v=64, dropout=dropout,
                        class_num=class_num).cuda()
    model = DataParallel(model, device_ids=[0, 1])
    chkpoint = torch.load(test_model_name)
    model.load_state_dict(chkpoint['model'])
    test_epoch(test_loader, device, model, test_data.__len__())