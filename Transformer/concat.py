import glob
import numpy as np
import pandas as pd

ent_num = '02'
save_dir = './' + ent_num + '/diff/'  
data_list = glob.glob(save_dir + 'diff_data*') 
label_list = glob.glob(save_dir + 'diff_label*') 
ori_data = np.load(save_dir + 'ori_data.npy')
ori_label = np.load(save_dir + 'ori_label.npy')

all_data = []
all_label = []
for file in data_list:
    data = np.load(file)
    all_data.append(data)

for file in label_list:
    label = np.load(file)
    all_label.append(label)

all_data = np.concatenate(all_data, axis=0)
all_label = np.concatenate(all_label, axis=0)

result_data = np.concatenate([ori_data, all_data])
result_label = np.concatenate([ori_label, all_label])
a = pd.DataFrame(result_label)

raw_train = np.c_[result_data, result_label]
np.random.shuffle(raw_train)
data, label = raw_train[:, :-1], raw_train[:, -1]

np.save('./' + ent_num + '/data/diff_data', data) 
np.save('./' + ent_num + '/data/diff_label', label) 