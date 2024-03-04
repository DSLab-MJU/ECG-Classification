from torch.utils.data import Dataset
import torch
import numpy as np
import os
from random import shuffle
import random
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import minmax_scale
from data_ import filter_data, random_10s_beat_align, aligned

seed = 42
ent_num = '02'
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

class my_dataset(Dataset):
    def __init__(self, path=os.getcwd() + '/data/', train=False, test=False, eval=False):
        self.path = path
        self.train = os.listdir(path)
        self.train = [i.split('_')[0] for i in self.train]
        self.train = list(set(self.train))
        self.train.sort()
        self.test = test
        self.train = self.train
        self.data = []
        self.label = []
        self.eval = eval
        max_R = 169
        max_len = 500

        if train:
            for index in self.train:
                data = np.load(self.path + '{}_data.npy'.format(index))
                data = filter_data(data, average=True)
                data = data.reshape(1, -1)
                label = np.load(self.path + '{}_detect_label.npy'.format(index))
                self.data.append(data)

                label = [i for i in label if i[1] - i[0] < 1400 and i[-1] != 4 and i[1] - i[0] != 1175]
                for j in range(len(label)):
                    start = label[j][0]
                    end = label[j][1]
                    R_loc = label[j][2]
                    if end - start > 250:
                        if R_loc - start > 100:
                            start = R_loc - 100
                            if end - R_loc > 200:
                                end = R_loc + 200
                        elif end - R_loc > 200:
                            end = R_loc + 200
                    label[j][0] = start
                    label[j][1] = end
                    label[j][2] = R_loc - start
                self.label.append(label)

            tmp_data1, tmp_label1 = random_10s_beat_align(self)
            tmp_data1, tmp_label1 = aligned(tmp_data1, tmp_label1, max_R, max_len)
            self.data = tmp_data1.copy()
            self.label = tmp_label1.copy()

            tmp_data = []
            for i in self.data:
                i = [ii.transpose() for ii in i]
                tmp_data.extend(i)

            self.data = tmp_data
            self.data = [i.reshape(-1) for i in self.data]
            index_normal = [i for i in range(len(self.label)) if self.label[i] == 0]
            shuffle(index_normal)
            index2 = [i for i in range(len(self.label)) if self.label[i] != 0]
            index3 = index_normal[:int(0.15 * len(index_normal))]
            index3.extend(index2)
            self.data = [self.data[i] for i in index3]
            self.label = [self.label[i] for i in index3]
            self.data = np.asarray(self.data)
            self.label = np.asarray(self.label)

            set_seed(seed)
            train_data, test_data, train_label, test_label = train_test_split(self.data, self.label, test_size = 0.2, random_state = seed) 
            eval_data, test_data, eval_label, test_label = train_test_split(test_data, test_label, test_size = 0.5, random_state = seed) 
            
            train_data = norm_data2(train_data)
            eval_data = norm_data2(eval_data)
            test_data = norm_data2(test_data)

            np.save('./' + ent_num + '/data/back_test_data', test_data) 
            np.save('./' + ent_num + '/data//back_test_label', test_label)

            np.save('./' + ent_num + '/data/back_eval_data', eval_data) 
            np.save('./' + ent_num + '/data/back_eval_label', eval_label)

            np.save('./' + ent_num + '/data/back_train_data', train_data) 
            np.save('./' + ent_num + '/data/back_train_label', train_label)

            self.data = np.load('./' + ent_num + '/data/diff_data.npy')
            self.label = np.load('./' + ent_num + '/data/diff_label.npy')
    
        elif test:
            self.data = np.load('./' + ent_num + '/data/back_test_data.npy')
            self.label = np.load('./' + ent_num + '/data/back_test_label.npy')

        elif eval:
            self.data = np.load('./' + ent_num + '/data/back_eval_data.npy')
            self.label = np.load('./' + ent_num + '/data/back_eval_label.npy')

        self.data = [self.data[i].reshape(500, 1) for i in range(len(self.data))]
        self.label = [self.label[i] for i in range(len(self.label))]

    def __getitem__(self, item):
        tmp_label = self.label[item]
        tmp_data = self.data[item]
        return tmp_data, tmp_label

    def __len__(self):
        return len(self.data)
