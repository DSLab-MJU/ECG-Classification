from data_corrector import corrector
from torch.utils.data import Dataset
import random
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

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

class loader(Dataset):
    def __init__(self, path=None, params=None, sp = 1):
        mode = params['mode']
        a = corrector()
        a.process(path)
        self.data = a.data
        self.label = a.label
        self.window = a.time
        self.r_peak = a.r_peak
        total = len(self.label)


        if mode == 'training':
            if sp == 1: # train1
                set_seed(seed)
                train_data, test_data, train_label, test_label, train_window, test_window, train_r_peak, test_r_peak = train_test_split(self.data, self.label, self.window, self.r_peak, test_size=0.2, random_state=seed)
                train_data = norm_data2(train_data)
            
                with open("./" + ent_num + "/data/seg_train_data.pkl","wb") as f:
                    pickle.dump(train_data, f)
                with open("./" + ent_num + "/data/seg_train_label.pkl","wb") as f:
                    pickle.dump(train_label, f)
                with open("./" + ent_num + "/data/seg_train_window.pkl","wb") as f:
                    pickle.dump(train_window, f)
                with open("./" + ent_num + "/data/seg_train_r_peak.pkl","wb") as f:
                    pickle.dump(train_r_peak, f)

                set_seed(seed)
                eval_data, test_data, eval_label, test_label, eval_window, test_window, eval_r_peak, test_r_peak = train_test_split(test_data, test_label, test_window, test_r_peak, test_size=0.5, random_state=seed)
                eval_data = norm_data2(eval_data)
                test_data = norm_data2(test_data)
                with open("./" + ent_num + "/data/seg_eval_data.pkl","wb") as f:
                    pickle.dump(eval_data, f)
                with open("./" + ent_num + "/data/seg_eval_label.pkl","wb") as f:
                    pickle.dump(eval_label, f)
                with open("./" + ent_num + "/data/seg_eval_window.pkl","wb") as f:
                    pickle.dump(eval_window, f)
                with open("./" + ent_num + "/data/seg_eval_r_peak.pkl","wb") as f:
                    pickle.dump(eval_r_peak, f)


                with open("./" + ent_num + "/data/seg_test_data.pkl","wb") as f:
                    pickle.dump(test_data, f)
                with open("./" + ent_num + "/data/seg_test_label.pkl","wb") as f:
                    pickle.dump(test_label, f)
                with open("./" + ent_num + "/data/seg_test_window.pkl","wb") as f:
                    pickle.dump(test_window, f)
                with open("./" + ent_num + "/data/seg_test_r_peak.pkl","wb") as f:
                    pickle.dump(test_r_peak, f)
                
                self.data = train_data
                self.label = train_label
                self.window = train_window
                self.r_peak = train_r_peak

            else: # train2
                with open("./" + ent_num + "/data/seg_train_data.pkl","rb") as f:
                    self.data = pickle.load(f)
                with open("./" + ent_num + "/data/seg_train_label.pkl","rb") as f:
                    self.label = pickle.load(f)
                with open("./" + ent_num + "/data/seg_train_window.pkl","rb") as f:
                    self.window = pickle.load(f)
                with open("./" + ent_num + "/data/seg_train_r_peak.pkl","rb") as f:
                    self.r_peak = pickle.load(f) 


        elif mode == 'test':
            with open("./" + ent_num + "/data/seg_test_data.pkl","rb") as f:
                self.data = pickle.load(f)
            with open("./" + ent_num + "/data/seg_test_label.pkl","rb") as f:
                self.label = pickle.load(f)
            with open("./" + ent_num + "/data/seg_test_window.pkl","rb") as f:
                self.window = pickle.load(f)
            with open("./" + ent_num + "/data/seg_test_r_peak.pkl","rb") as f:
                self.r_peak = pickle.load(f) 


        elif mode == 'eval':
            with open("./" + ent_num + "/data/seg_eval_data.pkl","rb") as f:
                self.data = pickle.load(f)
            with open("./" + ent_num + "/data/seg_eval_label.pkl","rb") as f:
                self.label = pickle.load(f)
            with open("./" + ent_num + "/data/seg_eval_window.pkl","rb") as f:
                self.window = pickle.load(f)
            with open("./" + ent_num + "/data/seg_eval_r_peak.pkl","rb") as f:
                self.r_peak = pickle.load(f) 


    def __getitem__(self, item):
        return self.data[item], self.window[item], self.label[item], self.r_peak[item], item

    def __len__(self):
        return len(self.data)
